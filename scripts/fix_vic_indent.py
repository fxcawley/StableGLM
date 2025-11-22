def fix_vic():
    filepath = "rashomon/rashomon_set.py"
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Dedent lines 1248 to 1338 by 4 spaces
    # Line numbers from grep (1-based) match python enumerate + 1
    
    start = 1248
    end = 1338
    
    new_lines = []
    for i, line in enumerate(lines):
        line_num = i + 1
        if start <= line_num < end:
            if line.startswith("    "):
                new_lines.append(line[4:])
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)
            
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Fixed VIC indentation.")

if __name__ == "__main__":
    fix_vic()

