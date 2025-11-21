import os

def fix_indentation():
    filepath = "rashomon/rashomon_set.py"
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    fixed = False
    for i, line in enumerate(lines):
        # Fix 1038
        if "else:" in lines[i-1] and "rsnew = float(r @ r)" in line:
            # Check if it needs more indentation
            # It should have 16 spaces, currently likely has 12
            if not line.startswith("                "):
                print(f"Fixing line {i+1}: {line.strip()}")
                lines[i] = "                rsnew = float(r @ r)\n"
                fixed = True

    if fixed:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        print("File updated.")
    else:
        print("No changes needed or pattern not found.")

if __name__ == "__main__":
    fix_indentation()

