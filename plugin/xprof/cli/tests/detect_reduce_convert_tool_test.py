"""Tests for the detect_reduce_convert_tool HLO parser suggestions."""

import json
from absl.testing import absltest
from xprof.cli.tools import detect_reduce_convert_tool


class DetectReduceConvertToolTest(absltest.TestCase):

  def test_no_inefficient_ops(self):
    def mock_get_top_ops(session_id, limit):
      del session_id, limit
      return json.dumps({"top_by_time": [], "top_by_bytes_accessed": []})

    result_json = detect_reduce_convert_tool.detect_reduce_convert(
        "session_123",
        get_top_hlo_ops_fn=mock_get_top_ops,
    )
    result = json.loads(result_json)
    self.assertFalse(result["bottlenecks_found"])

  def test_trace_upcast_upstream(self):
    # Setup mock computation mapping
    comp_info = {
        "params": {"param_0": "bf16"},
        "instrs": {
            "convert.1": {
                "name": "convert.1",
                "opcode": "convert",
                "type": "f32",
                "operands": ["param_0"],
                "line": "%convert.1 = f32[] convert(%param_0)",
            },
            "abs.1": {
                "name": "abs.1",
                "opcode": "abs",
                "type": "f32",
                "operands": ["convert.1"],
                "line": "%abs.1 = f32[] abs(%convert.1)",
            },
        },
    }
    # Trace upstream from abs.1
    res = detect_reduce_convert_tool._trace_upcast_upstream(comp_info, "abs.1")
    self.assertIsNotNone(res)
    self.assertEqual(res["type"], "upcast")
    self.assertEqual(res["instr"]["name"], "convert.1")

    # Trace upstream from param_0 directly
    res = detect_reduce_convert_tool._trace_upcast_upstream(
        comp_info, "param_0"
    )
    self.assertIsNotNone(res)
    self.assertEqual(res["type"], "parameter")
    self.assertEqual(res["name"], "param_0")

  def test_parse_multiple_computations(self):
    hlo = """
    ENTRY my_entry_comp (param_0: bf16[100]) -> bf16[100] {
      %param_0 = parameter(0)
      ROOT %fusion.1 = bf16[100] fusion(%param_0), calls=my_fusion
    }

    %my_fusion (param_0.1: bf16[100]) -> bf16[100] {
      %param_0.1 = parameter(0)
      %convert.1 = f32[100] convert(%param_0.1)
      ROOT %convert.2 = bf16[100] convert(%convert.1)
    }
    """
    computations, _, entry = detect_reduce_convert_tool._parse_hlo(hlo)
    self.assertEqual(entry, "my_entry_comp")
    self.assertIn("my_entry_comp", computations)
    self.assertIn("my_fusion", computations)

  def test_detect_reduce_convert_integration(self):
    # Setup mock content for HLO module
    hlo_content = """
    ENTRY my_entry_comp (param_0: bf16[100]) -> bf16[100] {
      %param_0 = parameter(0)
      ROOT %fusion.1 = bf16[100] fusion(%param_0), calls=my_fusion
    }

    %my_fusion (param_0.1: bf16[100]) -> bf16[100] {
      %param_0.1 = parameter(0)
      %convert.1 = f32[100] convert(%param_0.1)
      %abs.1 = f32[100] abs(%convert.1)
      %reduce.1 = f32[100] reduce(%abs.1, %abs.1), to_apply=add_comp
      ROOT %convert.2 = bf16[100] convert(%reduce.1)
    }
    """

    def mock_get_top_ops(session_id, limit):
      del session_id, limit
      return json.dumps({
          "top_by_time": [{
              "name": "by_program/jit_my_entry_comp/fusion.1",
              "total_self_time_ms": 10.0,
          }],
          "top_by_bytes_accessed": [],
      })

    def mock_get_hlo_content(session_id, module_name, max_lines):
      del session_id, module_name, max_lines
      return hlo_content

    result_json = detect_reduce_convert_tool.detect_reduce_convert(
        "session_123",
        get_top_hlo_ops_fn=mock_get_top_ops,
        get_hlo_module_content_fn=mock_get_hlo_content,
    )
    result = json.loads(result_json)
    if not result.get("bottlenecks_found"):
      print(f"DEBUG: result = {result_json}")
    self.assertTrue(result["bottlenecks_found"])
    self.assertLen(result["inefficient_ops"], 1)
    self.assertIn(
        "Detected unnecessary promotion pattern",
        result["inefficient_ops"][0]["recommendation"],
    )

  def test_detect_reduce_convert_mismatched_param_names(self):
    # Setup HLO where parameter variables inside the body do not match
    # signature names

    hlo_content = """
    ENTRY my_entry_comp (param_0: bf16[100]) -> bf16[100] {
      %p0 = parameter(0)
      ROOT %fusion.1 = bf16[100] fusion(%p0), calls=my_fusion
    }

    %my_fusion (Arg_0: bf16[100]) -> bf16[100] {
      %param_0 = parameter(0)
      %convert.1 = f32[100] convert(%param_0)
      %abs.1 = f32[100] abs(%convert.1)
      %reduce.1 = f32[100] reduce(%abs.1, %abs.1), to_apply=add_comp
      ROOT %convert.2 = bf16[100] convert(%reduce.1)
    }
    """

    def mock_get_top_ops(session_id, limit):
      del session_id, limit
      return json.dumps({
          "top_by_time": [{
              "name": "by_program/jit_my_entry_comp/fusion.1",
              "total_self_time_ms": 10.0,
          }],
          "top_by_bytes_accessed": [],
      })

    def mock_get_hlo_content(session_id, module_name, max_lines):
      del session_id, module_name, max_lines
      return hlo_content

    result_json = detect_reduce_convert_tool.detect_reduce_convert(
        "session_123",
        get_top_hlo_ops_fn=mock_get_top_ops,
        get_hlo_module_content_fn=mock_get_hlo_content,
    )
    result = json.loads(result_json)
    self.assertTrue(
        result["bottlenecks_found"],
        "Failed to detect bottleneck due to mismatched parameter names!"
        f" Result: {result_json}",
    )


if __name__ == "__main__":
  absltest.main()
