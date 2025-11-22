def fix_indentation():
    filepath = "rashomon/rashomon_set.py"
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Ranges are 1-based inclusive from my notes, need to convert to 0-based exclusive
    # But I'll use the line numbers from the file directly.
    # Note: line 1348 in file is index 1347.
    
    ranges = [
        (1348, 1387),
        (1400, 1435),
        (1448, 1501), # 1500 + 1
        (1504, 1508),
        (1520, 1662)
    ]
    
    new_lines = []
    for i, line in enumerate(lines):
        line_num = i + 1
        
        # Check if in any body range
        in_body = False
        for start, end in ranges:
            if start <= line_num < end:
                in_body = True
                break
        
        if in_body:
            new_lines.append("    " + line)
        else:
            new_lines.append(line)
            
    # Now handling the nested loop in MCR
    # It is in range 1603 to 1647.
    # Since we just indented 1520-1662, these lines are now indented by 4.
    # We need to indent them AGAIN by 4 to fix the logic bug.
    
    final_lines = []
    for i, line in enumerate(new_lines):
        line_num = i + 1
        if 1603 <= line_num <= 1647:
             final_lines.append("    " + line)
        else:
             final_lines.append(line)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(final_lines)
    print("Fixed all indentation.")

if __name__ == "__main__":
    fix_indentation()

