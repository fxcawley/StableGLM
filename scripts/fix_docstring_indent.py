def fix_docstrings():
    filepath = "rashomon/rashomon_set.py"
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    ranges = [
        (1339, 1348),
        (1389, 1400),
        (1437, 1448),
        (1510, 1520)
    ]
    
    new_lines = []
    for i, line in enumerate(lines):
        line_num = i + 1
        
        in_range = False
        for start, end in ranges:
            if start <= line_num < end:
                in_range = True
                break
        
        if in_range:
            new_lines.append("    " + line)
        else:
            new_lines.append(line)
            
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Fixed docstring/arg indentation.")

if __name__ == "__main__":
    fix_docstrings()

