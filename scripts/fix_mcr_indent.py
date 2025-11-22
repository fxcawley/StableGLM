def fix_indent():
    filepath = "rashomon/rashomon_set.py"
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    in_loop = False
    for i, line in enumerate(lines):
        # 0-based index. Line 1603 in file is index 1602.
        # But line numbers in read_file output might be slightly different if file changed.
        # I'll match the content "for j in range(self._d):"
        
        if "for j in range(self._d):" in line and "        for j" in line: # Match 8 spaces
            print(f"Found loop start at line {i+1}")
            in_loop = True
            new_lines.append("    " + line) # Add 4 spaces
        elif in_loop:
            # Check if we hit the end
            # The line AFTER the loop body is "# Aggregate across samples" or "mean_importance ="
            # line 1648 in previous read
            if "# Aggregate across samples" in line:
                in_loop = False
                new_lines.append(line)
            else:
                new_lines.append("    " + line)
        else:
            new_lines.append(line)
            
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Fixed indentation.")

if __name__ == "__main__":
    fix_indent()

