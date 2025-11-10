import sys
sys.path.insert(0, r'E:\Python\课设摸鱼')

try:
    from complete_pipeline import DepthCameraProcessor
    print("Module imported successfully")
except Exception as e:
    import traceback
    print("Import failed:")
    traceback.print_exc()
    
    # 尝试定位具体问题
    import ast
    with open(r'E:\Python\课设摸鱼\complete_pipeline.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        tree = ast.parse(content)
        print("AST parsing successful")
        
        # 查找可能的问题
        import re
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'import os' in line and 'def ' in '\n'.join(lines[:i]):  # 如果在函数定义之后
                func_before = False
                for j in range(i-1, -1, -1):
                    if j < len(lines) and lines[j].strip().startswith('def '):
                        func_before = True
                        break
                    elif j < len(lines) and lines[j].strip().startswith('class '):
                        break
                if func_before:
                    print(f"Potential issue at line {i}: {line.strip()}")
                    
    except SyntaxError as se:
        print(f"Syntax error at line {se.lineno}: {se.msg}")