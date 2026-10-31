"""Show the range explosion that makes gravitational unfindable."""
import sympy as sp
from purediscover.benchmark.catalog import get_law, make_data

for name in ("coulomb", "gravitational"):
    law = get_law(name)
    data = make_data(law, 40, 0.02)
    ys = [y for _, y in data]
    print(f"\n{name}:")
    print(f"  var ranges: {law.var_ranges}")
    print(f"  y range: {min(ys):.3e} .. {max(ys):.3e}")
    print(f"  decades spanned: {__import__('math').log10(max(ys)/min(ys)):.1f}")
