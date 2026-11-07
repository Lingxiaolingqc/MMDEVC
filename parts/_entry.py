"""Forward part-specific commands to the verified project scripts."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def dispatch(title, commands):
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(title)
        print("Usage: python " + str(Path(sys.argv[0]).relative_to(ROOT) if Path(sys.argv[0]).is_absolute() else sys.argv[0]) + " COMMAND [original script arguments]")
        for name, (script, fixed) in commands.items():
            print(f"  {name:16s} -> scripts/{script} {' '.join(fixed)}")
        print("Use COMMAND --help to inspect the original options. Working directory is the repository root.")
        return 0
    if args[0] not in commands:
        print(f"Unknown command: {args[0]}", file=sys.stderr)
        return 2
    script, fixed = commands[args[0]]
    if "--model" in fixed and any(a == "--model" or a.startswith("--model=") for a in args[1:]):
        print("This command fixes the model; choose the appropriate part command.", file=sys.stderr)
        return 2
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *fixed, *args[1:]], cwd=ROOT).returncode
