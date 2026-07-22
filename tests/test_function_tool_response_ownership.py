import ast
import unittest
from pathlib import Path


class FunctionToolResponseOwnershipTest(unittest.TestCase):
    def test_function_tools_never_request_an_automatic_follow_up(self):
        tree = ast.parse(
            (Path(__file__).parents[1] / "agent_v2.py").read_text(encoding="utf-8")
        )
        tools = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(
                (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Name)
                    and decorator.func.id == "function_tool"
                )
                or (
                    isinstance(decorator, ast.Name)
                    and decorator.id == "function_tool"
                )
                for decorator in node.decorator_list
            )
        ]

        self.assertTrue(tools, "agent_v2.py must expose function tools")
        non_none_returns = {
            tool.name: [
                ast.unparse(node.value)
                for node in ast.walk(tool)
                if isinstance(node, ast.Return)
                and node.value is not None
                and not (
                    isinstance(node.value, ast.Constant) and node.value.value is None
                )
            ]
            for tool in tools
        }
        self.assertEqual(
            {name: values for name, values in non_none_returns.items() if values},
            {},
            "Function tools explicitly own their replies, so returning a value would "
            "also ask LiveKit to generate an automatic follow-up",
        )


if __name__ == "__main__":
    unittest.main()
