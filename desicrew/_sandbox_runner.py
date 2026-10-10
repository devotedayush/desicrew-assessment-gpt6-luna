"""Trusted standalone worker. Never import or execute generated modules here."""

from __future__ import annotations

import ast
import json
import math
import sys

MAX_CODE_CHARS = 12_000
MAX_RESULT_ROWS = 200
MAX_RESULT_CHARS = 100_000
SAFE_FUNCTIONS = {"len", "min", "max", "sum", "round", "abs", "sorted", "str", "int", "float", "bool", "list", "dict", "tuple", "set"}
PANDAS_FUNCTIONS = {"DataFrame", "Series", "concat", "merge", "crosstab", "to_numeric", "to_datetime", "Timestamp", "isna", "notna"}
SAFE_METHODS = {
    "head", "tail", "nlargest", "nsmallest", "sum", "mean", "median", "min", "max", "count", "size", "nunique",
    "value_counts", "sort_values", "sort_index", "groupby", "agg", "aggregate", "reset_index", "rename", "fillna",
    "dropna", "isna", "notna", "duplicated", "drop_duplicates", "select_dtypes", "copy", "describe", "astype",
    "round", "abs", "clip", "cumsum", "cumprod", "cummax", "cummin", "rank", "isin", "between", "contains",
    "lower", "upper", "strip", "startswith", "endswith", "replace", "to_dict", "to_list", "tolist", "to_frame",
    "get", "drop", "merge", "join", "unique", "all", "any", "std", "var", "prod", "first", "last", "keys",
}
SAFE_ATTRIBUTES = SAFE_METHODS | PANDAS_FUNCTIONS | {
    "loc", "iloc", "at", "iat", "str", "dt", "columns", "index", "shape", "empty", "dtypes", "ndim", "name",
    "values", "T", "year", "month", "day", "date", "hour", "minute", "dayofweek", "quarter",
}
AGGREGATIONS = {"sum", "mean", "median", "min", "max", "count", "size", "nunique", "std", "var", "first", "last", "prod", "any", "all"}
DTYPES = {"int", "int64", "int32", "Int64", "Int32", "float", "float64", "float32", "Float64", "str", "string", "object", "bool", "boolean", "category"}
SAFE_NODES = (
    ast.Module, ast.Assign, ast.Expr, ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.Subscript, ast.Slice, ast.Attribute, ast.Call, ast.keyword, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.IfExp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr, ast.BitXor,
    ast.USub, ast.UAdd, ast.Not, ast.Invert, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot,
)


class SandboxValidationError(ValueError):
    """Code is outside the intentionally small query language."""


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        raise SandboxValidationError("This argument must be a literal, not executable code.") from None


def _check_aggregation(value, named=False):
    if isinstance(value, str) and value in AGGREGATIONS:
        return
    if isinstance(value, (list, tuple)):
        if named and len(value) == 2 and isinstance(value[0], str) and value[1] in AGGREGATIONS:
            return
        for part in value:
            _check_aggregation(part)
        return
    if isinstance(value, dict):
        for part in value.values():
            _check_aggregation(part)
        return
    raise SandboxValidationError("Aggregation functions must be approved statistical reductions.")


def validate_code(code: str) -> None:
    """Allow pandas expressions, rejecting reflection, imports, callbacks and I/O."""
    if not isinstance(code, str) or not code.strip():
        raise SandboxValidationError("The query must contain Python code.")
    if len(code) > MAX_CODE_CHARS:
        raise SandboxValidationError("Generated code exceeds the size limit.")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxValidationError(f"Invalid Python syntax on line {exc.lineno}.") from None
    nodes = list(ast.walk(tree))
    if len(nodes) > 1_500:
        raise SandboxValidationError("Generated code is too complex.")
    if not any(isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets) for node in nodes):
        raise SandboxValidationError("Assign the final answer to result.")
    for node in nodes:
        if not isinstance(node, SAFE_NODES):
            raise SandboxValidationError(f"Unsupported operation: {type(node).__name__}.")
        if isinstance(node, ast.Name) and (node.id.startswith("_") or "__" in node.id):
            raise SandboxValidationError("Private names are forbidden.")
        if isinstance(node, ast.Assign):
            if any(not isinstance(target, ast.Name) for target in node.targets):
                raise SandboxValidationError("Only local variable assignments are permitted.")
            if any(target.id in {"pd", "sheets"} or target.id in SAFE_FUNCTIONS for target in node.targets):
                raise SandboxValidationError("Cannot replace trusted query bindings.")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in SAFE_ATTRIBUTES:
                raise SandboxValidationError(f"Attribute is not allowed: {node.attr}.")
            if isinstance(node.value, ast.Name) and node.value.id == "pd" and node.attr not in PANDAS_FUNCTIONS:
                raise SandboxValidationError(f"pandas access is not allowed: {node.attr}.")
        if isinstance(node, ast.Call):
            if any(isinstance(arg, ast.Starred) for arg in node.args) or any(kw.arg is None for kw in node.keywords):
                raise SandboxValidationError("Expanded arguments are forbidden.")
            if isinstance(node.func, ast.Name):
                if node.func.id not in SAFE_FUNCTIONS:
                    raise SandboxValidationError(f"Function is not allowed: {node.func.id}.")
            elif isinstance(node.func, ast.Attribute):
                method = node.func.attr
                is_pandas = isinstance(node.func.value, ast.Name) and node.func.value.id == "pd"
                if method not in (PANDAS_FUNCTIONS if is_pandas else SAFE_METHODS):
                    raise SandboxValidationError(f"Method is not allowed: {method}.")
                if method in {"agg", "aggregate"}:
                    for arg in node.args:
                        _check_aggregation(_literal(arg))
                    for kw in node.keywords:
                        if kw.arg not in {"axis", "numeric_only"}:
                            _check_aggregation(_literal(kw.value), named=True)
                if method == "astype":
                    dtype = _literal(node.args[0]) if node.args else _literal(next((kw.value for kw in node.keywords if kw.arg == "dtype"), ast.Constant(None)))
                    types = dtype.values() if isinstance(dtype, dict) else [dtype]
                    if any(not isinstance(value, str) or value not in DTYPES for value in types):
                        raise SandboxValidationError("Only approved, built-in pandas dtypes are allowed.")
                if method in {"DataFrame", "Series"}:
                    for kw in node.keywords:
                        if kw.arg == "dtype" and _literal(kw.value) not in DTYPES:
                            raise SandboxValidationError("Only approved pandas dtypes are allowed.")
            else:
                raise SandboxValidationError("Indirect function calls are forbidden.")


def _install_seccomp():
    """Fail closed. Deny filesystem access, networking and new processes/threads."""
    import ctypes
    import errno
    import os
    if sys.platform != "linux":
        raise RuntimeError("Secure query execution requires Linux with libseccomp.")
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.restype = None
    context = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("Cannot initialize the Linux query sandbox.")
    deny = 0x00050000 | errno.EPERM  # SCMP_ACT_ERRNO
    denied = {
        "open", "openat", "openat2", "creat", "readlink", "readlinkat", "getdents", "getdents64",
        "socket", "socketpair", "connect", "bind", "listen", "accept", "accept4", "sendto", "sendmsg", "sendmmsg",
        "clone", "clone3", "fork", "vfork", "execve", "execveat", "ptrace", "process_vm_readv", "process_vm_writev",
        "unlink", "unlinkat", "rename", "renameat", "renameat2", "mkdir", "mkdirat", "rmdir", "link", "linkat",
        "symlink", "symlinkat", "mount", "umount2", "chroot", "pivot_root", "chmod", "fchmod", "fchmodat",
        "truncate", "ftruncate", "mknod", "mknodat", "io_uring_setup", "bpf", "kill", "tkill", "tgkill",
    }
    try:
        for name in sorted(denied):
            number = lib.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number >= 0 and lib.seccomp_rule_add(context, deny, number, 0) != 0:
                raise RuntimeError(f"Cannot apply sandbox rule: {name}.")
        if lib.seccomp_load(context) != 0:
            raise RuntimeError("Cannot enable the Linux query sandbox.")
    finally:
        lib.seccomp_release(context)


def _json_value(value, truncation, depth=0):
    """Only serialize simple, bounded data. No repr of arbitrary objects."""
    import datetime
    import numpy as np
    import pandas as pd
    if depth > 8:
        truncation.add("depth_limit")
        return "[depth limit]"
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > 2_000:
            truncation.add("string_limit")
        return value[:2_000] if isinstance(value, str) else value
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (datetime.datetime, datetime.date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, dict):
        if len(value) > MAX_RESULT_ROWS:
            truncation.add("collection_limit")
        output = {}
        for key, part in list(value.items())[:MAX_RESULT_ROWS]:
            name = str(key)
            if len(name) > 200:
                truncation.add("key_limit")
            bounded_name = name[:200]
            if bounded_name in output:
                # Distinct keys can coincide after stringification or clipping.
                truncation.add("key_collision")
            output[bounded_name] = _json_value(part, truncation, depth + 1)
        return output
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        if len(value) > MAX_RESULT_ROWS:
            truncation.add("collection_limit")
        return [_json_value(part, truncation, depth + 1) for part in list(value)[:MAX_RESULT_ROWS]]
    raise TypeError(f"Unsupported result type: {type(value).__name__}.")


def _serialize_result(value):
    import pandas as pd
    truncation = set()
    if isinstance(value, pd.Series):
        value = value.rename(value.name or "value").reset_index()
    if isinstance(value, pd.DataFrame):
        # Grouping labels and describe() statistics often live in the index.
        # Preserve them rather than displaying anonymous rows with lost meaning.
        if not isinstance(value.index, pd.RangeIndex) or any(name is not None for name in value.index.names):
            names = []
            for position, name in enumerate(value.index.names):
                candidate = str(name) if name is not None else ("row_label" if value.index.nlevels == 1 else f"row_label_{position + 1}")
                while candidate in value.columns or candidate in names:
                    candidate = "index_" + candidate
                names.append(candidate)
            value = value.rename_axis(names).reset_index()
        if len(value.columns) > 100:
            raise ValueError("Result has more than 100 columns; select fewer columns.")
        if len(value) > MAX_RESULT_ROWS:
            truncation.add("row_limit")
        result = {"type": "table", "columns": [str(column) for column in value.columns], "rows": _json_value(value.head(MAX_RESULT_ROWS).to_dict(orient="records"), truncation), "row_count": len(value)}
    elif isinstance(value, dict):
        result = {"type": "object", "value": _json_value(value, truncation)}
    elif isinstance(value, (list, tuple, set)):
        result = {"type": "list", "items": _json_value(value, truncation), "item_count": len(value)}
    else:
        result = {"type": "scalar", "value": _json_value(value, truncation)}
    # Completeness applies recursively, not just to the outer object or row count.
    result["truncated"] = bool(truncation)
    result["complete"] = not bool(truncation)
    result["truncation_reasons"] = sorted(truncation)
    return result


def main():
    import builtins
    import io
    import os
    import resource
    import pandas as pd
    # Trusted imports and dataframe reconstruction happen before denying all file opens.
    payload = json.loads(sys.stdin.buffer.read(32 * 1024 * 1024 + 1))
    validate_code(payload["code"])
    sheets = {name: pd.read_json(io.StringIO(encoded), orient="table") for name, encoded in payload["sheets"].items()}
    if not sheets:
        raise ValueError("No worksheet was supplied.")
    # Warm ordinary pandas paths before seccomp prevents lazy imports.
    warm = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    warm.groupby("b")["a"].sum().reset_index()
    warm.describe(include="all")
    warm.to_dict(orient="records")
    warm.nlargest(1, "a")
    warm.nsmallest(1, "a")
    warm.merge(warm, on="b")
    pd.crosstab(warm["b"], warm["a"])
    pd.to_numeric(warm["a"], errors="coerce")
    pd.to_datetime(pd.Series(["2024-01-01"]), errors="coerce")
    warm["b"].str.contains("x", regex=False)
    code = compile(payload["code"], "<inventory-query>", "exec")
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
    cpu = max(1, min(int(payload.get("cpu_seconds", 3)), 10))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if hasattr(resource, "RLIMIT_NPROC"):
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    # Only anonymous pipes 0/1/2 remain; no inherited file or socket handles.
    os.closerange(3, 1024)
    _install_seccomp()
    scope = {"__builtins__": {name: getattr(builtins, name) for name in SAFE_FUNCTIONS}, "pd": pd, "sheets": sheets, "df": next(iter(sheets.values()))}
    exec(code, scope, scope)
    result = _serialize_result(scope["result"])
    output = json.dumps({"ok": True, "result": result, "isolation": "linux-seccomp"}, ensure_ascii=False, allow_nan=False)
    if len(output) > MAX_RESULT_CHARS:
        raise ValueError("Result exceeds the output limit; aggregate or select fewer rows.")
    print(output)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        # Exception names/messages are bounded; traceback paths and runtime state stay private.
        detail = "I/O operation denied by query sandbox." if isinstance(exc, PermissionError) else str(exc)[:500]
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {detail}"}, ensure_ascii=False))
        sys.exit(1)
