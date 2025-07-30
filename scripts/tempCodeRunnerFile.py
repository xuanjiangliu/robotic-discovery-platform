# --- Path Setup ---
# This is the standard way to make a script runnable from anywhere in the project
# and ensure it can find the 'pkg' directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)