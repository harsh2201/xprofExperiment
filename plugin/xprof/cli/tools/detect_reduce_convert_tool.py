"""MCP tool to detect unnecessary f32 promotions in reduction operations."""

import collections
from collections.abc import Callable
import json
import logging
import re

from xprof.cli.internal.oss import hlo_tools
from xprof.cli.tools import get_top_hlo_ops_tool


def _parse_hlo(hlo_text):
  """Parses raw HLO text into structured computations.

  Args:
    hlo_text: The raw HLO text.

  Returns:
    A tuple of (computations, flat_instrs, entry_comp) where computations is a
    dict of parsed computation data, flat_instrs maps instruction names to
    (computation_name, instruction_dict), and entry_comp is the name of the
    entry computation.
  """
  # comp_pattern matches:
  # Group 1: ENTRY prefix (optional)
  # Group 2: comp_name
  # Group 3: params block (e.g. "(param_0: type, ...)")
  # Group 4: body (inside {...})
  comp_pattern = r"((?:ENTRY\s+)?)(%?[a-zA-Z0-9._-]+)\s*(\(.*?\))\s*->\s*.*?{\n([\s\S]*?)^\s*}"
  computations = {}
  flat_instrs = {}
  entry_comp = None
  for match in re.finditer(comp_pattern, hlo_text, re.MULTILINE):
    is_entry = bool(match.group(1))
    comp_name = match.group(2).lstrip("%")
    params_str = match.group(3)
    comp_body = match.group(4)

    # Parse params
    params = {}
    param_pattern = r"(%?[a-zA-Z0-9._-]+):\s*([a-zA-Z0-9._-]+)\[.*?\]"
    for m in re.finditer(param_pattern, params_str):
      params[m.group(1).replace("%", "")] = m.group(2)

    instrs, root = _parse_instructions(comp_body)
    consumers_map = collections.defaultdict(list)
    for _, instr in instrs.items():
      for operand in instr["operands"]:
        consumers_map[operand].append(instr)

    computations[comp_name] = {
        "name": comp_name,
        "params": params,
        "instrs": instrs,
        "root": root,
        "consumers_map": consumers_map,
        "is_entry": is_entry,
    }
    for instr_name, instr in instrs.items():
      flat_instrs[instr_name] = (comp_name, instr)
    if is_entry:
      entry_comp = comp_name
  return computations, flat_instrs, entry_comp


def _parse_instructions(comp_body):
  """Parses instructions inside a computation body.

  Args:
    comp_body: The body text of the computation.

  Returns:
    A tuple of (instructions, root_instr) where instructions is a dict of
    parsed instructions and root_instr is the ROOT instruction if found.
  """
  instructions = {}
  root_instr = None
  for line in comp_body.splitlines():
    line_strip = line.strip()
    if (
        not line_strip
        or line_strip.startswith("}")
        or line_strip.startswith("//")
    ):
      continue

    # Extract ROOT prefix
    is_root = False
    if line_strip.startswith("ROOT "):
      is_root = True
      line_strip = line_strip[5:].strip()

    # Split name and expression
    if "=" not in line_strip:
      continue
    name_part, expr_part = line_strip.split("=", 1)
    name = name_part.replace("%", "").strip()

    # Match: type opcode(operands)rest (type is optional)
    match = re.match(
        r"^(?:(.*?)\s+)?([a-zA-Z0-9._-]+)\((.*?)\)(.*)$", expr_part.strip()
    )
    if match:
      type_part = match.group(1) or ""
      opcode = match.group(2)
      operands_str = match.group(3)
      rest = match.group(4)

      operands = [
          op.replace("%", "").strip()
          for op in operands_str.split(",")
          if op.strip()
      ]

      called_comp = None
      match_calls = re.search(r"calls=(%?[a-zA-Z0-9._-]+)", rest)
      if match_calls:
        called_comp = match_calls.group(1).lstrip("%")

      # Extract base type (e.g. f32[100] -> f32, or tuple -> raw tuple type)
      base_type = type_part
      if type_part and not type_part.startswith("("):
        base_type = type_part.split("[")[0]

      instr = {
          "name": name,
          "type": base_type,
          "opcode": opcode,
          "operands": operands,
          "called_comp": called_comp,
          "is_root": is_root,
          "line": line_strip,
      }
      instructions[name] = instr
      if is_root:
        root_instr = instr
  return instructions, root_instr


def _trace_upcast_upstream(comp_info, start_op_name, visited=None):
  """Traces upstream in the computation to find a convert op from bf16 to f32.

  Supports intermediate element-wise operations by traversing operands.

  Args:
    comp_info: Dict containing instruction data for the current computation.
    start_op_name: Name of the instruction to start tracing upstream from.
    visited: Set of visited instruction names to avoid cycles.

  Returns:
    A dict representing the found upcast (with type 'upcast' and 'instr'),
    or a parameter reference (with type 'parameter' and 'name'), or None.
  """
  if visited is None:
    visited = set()

  queue = collections.deque([start_op_name])
  while queue:
    current_op_name = queue.popleft()
    if current_op_name in visited:
      continue
    visited.add(current_op_name)

    if current_op_name in comp_info["params"]:
      return {"type": "parameter", "name": current_op_name}

    if current_op_name not in comp_info["instrs"]:
      continue

    instr = comp_info["instrs"][current_op_name]
    opcode = instr["opcode"]

    if opcode == "parameter":
      try:
        param_idx = int(instr["operands"][0])
        param_keys = list(comp_info["params"].keys())
        if 0 <= param_idx < len(param_keys):
          return {"type": "parameter", "name": param_keys[param_idx]}
      except (ValueError, IndexError):
        pass

    # Check if this instruction is a convert from bf16 to f32
    if opcode == "convert" and instr["type"] == "f32":
      operand = instr["operands"][0]
      op_type = None
      if (
          operand in comp_info["instrs"]
          and comp_info["instrs"][operand]["opcode"] == "parameter"
      ):
        try:
          param_idx = int(comp_info["instrs"][operand]["operands"][0])
          param_keys = list(comp_info["params"].keys())
          if 0 <= param_idx < len(param_keys):
            op_type = comp_info["params"][param_keys[param_idx]]
        except (ValueError, IndexError):
          pass
      if not op_type:
        if operand in comp_info["params"]:
          op_type = comp_info["params"][operand]
        elif operand in comp_info["instrs"]:
          op_type = comp_info["instrs"][operand]["type"]
      if op_type == "bf16":
        return {"type": "upcast", "instr": instr}

    # For element-wise or pass-through ops, trace upstream to operands
    # Skip constants to avoid unnecessary branches
    for operand in instr["operands"]:
      if (
          operand in comp_info["instrs"]
          and comp_info["instrs"][operand]["opcode"] == "constant"
      ):
        continue
      if operand not in visited:
        queue.append(operand)

  return None


def _trace_downcast(
    comp_info, start_op_name, computations, trace_logs, depth=0
):
  """Traces downcast transitively in the graph (outer graph or inside fusions).

  Args:
    comp_info: Dict of parsed computation info.
    start_op_name: The instruction name to start tracing from.
    computations: Dict of all parsed computations in the module.
    trace_logs: List to append log messages.
    depth: Current recursion depth (used for logging indentation).

  Returns:
    True if a downcast to bf16/f8 is found, False otherwise.
  """
  indent = "  " * (depth + 1)
  queue = collections.deque([start_op_name])
  visited = {start_op_name}

  while queue:
    curr_name = queue.popleft()
    consumers = comp_info["consumers_map"].get(curr_name, [])
    for consumer in consumers:
      if consumer["name"] not in visited:
        visited.add(consumer["name"])

        log_prefix = "  Tracing" if depth == 0 else "    Internal tracing"
        trace_logs.append(f"{log_prefix} consumer: {consumer['line']}")

        # Check if this consumer is a downcast
        if consumer["opcode"] == "convert" and (
            consumer["type"] == "bf16" or consumer["type"].startswith("f8")
        ):
          found_prefix = (
              "  FOUND standalone" if depth == 0 else "    Internal FOUND"
          )
          trace_logs.append(f"{found_prefix} downcast: {consumer['line']}")
          return True
        elif consumer["opcode"] == "fusion":
          called_comp = consumer["called_comp"]
          if called_comp in computations:
            fusion_data = computations[called_comp]
            fusion_params = fusion_data["params"]

            # Find parameter index corresponding to curr_name
            try:
              param_index = consumer["operands"].index(curr_name)
            except ValueError:
              continue

            # Find parameter name
            target_param = None
            for p_name in fusion_params:
              if p_name == f"param_{param_index}" or p_name.startswith(
                  f"param_{param_index}."
              ):
                target_param = p_name
                break

            if target_param:
              trace_logs.append(
                  f"{indent}Mapping to parameter {target_param} in"
                  f" {called_comp}"
              )
              # Recurse inside the fusion with depth + 1
              if _trace_downcast(
                  fusion_data, target_param, computations, trace_logs, depth + 1
              ):
                trace_logs.append(
                    f"{indent}FOUND downcast inside fusion {called_comp}"
                )
                return True
        queue.append(consumer["name"])
  return False


def detect_reduce_convert(
    session_id: str,
    get_top_hlo_ops_fn: Callable[
        ..., str
    ] = get_top_hlo_ops_tool.get_top_hlo_ops,
    get_hlo_module_content_fn: Callable[
        ..., str
    ] = hlo_tools.get_hlo_module_content,
    limit: int = 50,
) -> str:
  """Detects reduce ops that unnecessarily promote bf16 to f32.

  Args:
      session_id: The unique XProf session ID.
      get_top_hlo_ops_fn: Function to retrieve top HLO operations.
      get_hlo_module_content_fn: Function to retrieve HLO module content.
      limit: How many top operations to analyze.

  Returns:
      A JSON string summarizing the findings.
  """
  try:
    # 1. Fetch Top Ops
    top_ops_json = get_top_hlo_ops_fn(session_id, limit=limit)
    if not top_ops_json:
      return json.dumps({"error": "Could not fetch top HLO ops."})

    ops_data = json.loads(top_ops_json)
    top_by_bytes = ops_data.get("top_by_bytes_accessed", [])
    top_by_time = ops_data.get("top_by_time", [])

    # Combine and de-duplicate candidates
    unique_candidates = {}
    for op in top_by_time + top_by_bytes:
      name = op.get("name", "")
      if name not in unique_candidates:
        unique_candidates[name] = op

    # Sort by self time descending
    sorted_candidates = sorted(
        unique_candidates.values(),
        key=lambda x: x.get("total_self_time_ms", 0.0),
        reverse=True,
    )
    candidates = sorted_candidates

    inefficient_ops = []
    parsed_modules = {}  # cache: mod_name -> (computations, flat_instrs)

    for candidate in candidates:
      raw_name = candidate.get("name", "")
      parts = raw_name.split("/")
      if len(parts) <= 1 or "jit_" not in parts[1]:
        continue
      mod_name = parts[1]
      leaf_name = parts[-1].split(" and its ")[0].replace("%", "").strip()

      # Fetch and parse HLO if not already cached
      if mod_name not in parsed_modules:
        logging.info("Fetching HLO for module %s", mod_name)
        full_hlo = get_hlo_module_content_fn(
            session_id, module_name=mod_name, max_lines=-1
        )
        if not full_hlo:
          continue

        computations, flat_instrs, entry_comp = _parse_hlo(full_hlo)
        if not entry_comp:
          continue

        parsed_modules[mod_name] = (computations, flat_instrs)

      computations, flat_instrs = parsed_modules[mod_name]

      if leaf_name not in flat_instrs:
        continue

      found_in_comp, instr = flat_instrs[leaf_name]
      comp_info = computations[found_in_comp]

      is_inefficient = False
      trace_logs = []

      # Case 1: Candidate is a fusion instruction
      if instr["opcode"] == "fusion":
        called_comp = instr["called_comp"]
        if called_comp in computations:
          fusion_info = computations[called_comp]
          fusion_instrs = fusion_info["instrs"]
          fusion_root = fusion_info["root"]

          # Find reduction-like ops inside the fusion
          reduce_op = None
          convert_op = None
          for _, f_instr in fusion_instrs.items():
            if f_instr["opcode"] in {
                "reduce",
                "all-reduce",
                "all-reduce-start",
                "all-reduce-done",
            }:
              # Trace upstream from its operands
              for operand in f_instr["operands"]:
                upcast_res = _trace_upcast_upstream(fusion_info, operand)
                if upcast_res:
                  if upcast_res["type"] == "upcast":
                    convert_op = upcast_res["instr"]
                    reduce_op = f_instr
                    break
                  elif upcast_res["type"] == "parameter":
                    # Trace upstream in outer computation
                    try:
                      param_keys = list(fusion_info["params"].keys())
                      param_idx = param_keys.index(upcast_res["name"])
                      outer_operand = instr["operands"][param_idx]
                      outer_upcast_res = _trace_upcast_upstream(
                          comp_info, outer_operand
                      )
                      if (
                          outer_upcast_res
                          and outer_upcast_res["type"] == "upcast"
                      ):
                        convert_op = outer_upcast_res["instr"]
                        reduce_op = f_instr
                        break
                    except (ValueError, IndexError):
                      pass
              if reduce_op:
                break

          if reduce_op:
            trace_logs.append(
                f"Found upcast-reduce pattern involving fusion '{leaf_name}'"
                f" (calls {called_comp}):"
            )
            trace_logs.append(f"  Convert: {convert_op['line']}")
            trace_logs.append(f"  Reduce: {reduce_op['line']}")

            if _trace_downcast(
                fusion_info,
                reduce_op["name"],
                computations,
                trace_logs,
                depth=1,
            ):
              is_inefficient = True
            else:
              start_trace_op = leaf_name
              # If fusion returns a tuple, trace from GTE of reduce output
              if fusion_root and fusion_root["opcode"] == "tuple":
                try:
                  reduce_index = fusion_root["operands"].index(
                      reduce_op["name"]
                  )
                  trace_logs.append(
                      f"  Reduce output is at tuple index {reduce_index}"
                  )

                  # Find GTE in the same computation
                  gte_instr = None
                  for _, entry_instr in comp_info["instrs"].items():
                    if (
                        entry_instr["opcode"] == "get-tuple-element"
                        and entry_instr["operands"][0] == leaf_name
                    ):
                      match_index = re.search(
                          r"index=(\d+)", entry_instr["line"]
                      )
                      if (
                          match_index
                          and int(match_index.group(1)) == reduce_index
                      ):
                        gte_instr = entry_instr
                        break
                  if gte_instr:
                    trace_logs.append(f"  Found GTE: {gte_instr['line']}")
                    start_trace_op = gte_instr["name"]
                  else:
                    start_trace_op = None
                except ValueError:
                  start_trace_op = None

              if start_trace_op:
                trace_logs.append(
                    f"  Tracing downcast starting from '{start_trace_op}'..."
                )
                if _trace_downcast(
                    comp_info, start_trace_op, computations, trace_logs
                ):
                  is_inefficient = True

      # Case 2: Candidate is a reduction-like instruction itself
      elif instr["opcode"] in {
          "reduce",
          "all-reduce",
          "all-reduce-start",
          "all-reduce-done",
      }:
        convert_op = None
        for operand in instr["operands"]:
          upcast_res = _trace_upcast_upstream(comp_info, operand)
          if upcast_res and upcast_res["type"] == "upcast":
            convert_op = upcast_res["instr"]
            break

        if convert_op:
          trace_logs.append(
              "Found upcast-reduce pattern for standalone reduce"
              f" '{leaf_name}':"
          )
          trace_logs.append(f"  Convert: {convert_op['line']}")
          trace_logs.append(f"  Reduce: {instr['line']}")
          trace_logs.append(
              f"  Tracing downcast starting from '{leaf_name}'..."
          )
          if _trace_downcast(comp_info, leaf_name, computations, trace_logs):
            is_inefficient = True

      # Case 3: Candidate is an upcast convert instruction itself
      elif instr["opcode"] == "convert" and instr["type"] == "f32":
        convert_operand = instr["operands"][0]
        is_bf16 = False
        if convert_operand in comp_info["instrs"]:
          is_bf16 = comp_info["instrs"][convert_operand]["type"] == "bf16"
        elif convert_operand in comp_info["params"]:
          is_bf16 = comp_info["params"][convert_operand] == "bf16"

        if is_bf16:
          trace_logs.append(f"Found standalone upcast: {instr['line']}")
          # Find downstream reduction-like consumer
          reduce_consumers = []
          queue = collections.deque([leaf_name])
          visited = {leaf_name}
          while queue:
            curr = queue.popleft()
            consumers = comp_info["consumers_map"].get(curr, [])
            for consumer in consumers:
              if consumer["name"] not in visited:
                visited.add(consumer["name"])
                if consumer["opcode"] in {
                    "reduce",
                    "all-reduce",
                    "all-reduce-start",
                    "all-reduce-done",
                }:
                  reduce_consumers.append(consumer)
                elif (
                    consumer["type"] == "f32"
                    or consumer["type"].startswith("(")
                    or consumer["opcode"] in {"tuple", "get-tuple-element"}
                ):
                  queue.append(consumer["name"])

          for reduce_op in reduce_consumers:
            trace_logs.append(
                f"  Found reduction consumer: {reduce_op['line']}"
            )
            trace_logs.append(
                f"  Tracing downcast starting from '{reduce_op['name']}'..."
            )
            if _trace_downcast(
                comp_info, reduce_op["name"], computations, trace_logs
            ):
              is_inefficient = True
              break

      if is_inefficient:
        candidate["recommendation"] = (
            "Detected unnecessary promotion pattern (bf16 -> f32 -> reduce ->"
            f" bf16) involving '{leaf_name}' in module '{mod_name}'."
            " Consider running the reduction in bf16 by setting"
            " dtype=jnp.bfloat16 in your JAX reduction operator."
        )
        candidate["explanation"] = "\n".join(trace_logs)
        inefficient_ops.append(candidate)

    if inefficient_ops:
      message = (
          f"Detected {len(inefficient_ops)} reduction operations with potential"
          " default type promotion overhead."
      )
    else:
      message = "No inefficient reduction promotions detected."

    return json.dumps(
        {
            "bottlenecks_found": len(inefficient_ops) > 0,
            "inefficient_ops": inefficient_ops,
            "message": message,
        },
        indent=2,
    )

  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("Error detecting reduce convert overhead")
    return json.dumps({"error": f"Internal error during detection: {e}"})
