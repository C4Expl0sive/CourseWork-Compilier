from lexer_parser import *
from symbol_table import *
# from codegen import CodeGen
import traceback
# import rply
# from ast import *

fname = "input.toy"
with open(fname) as f:
    text_input = f.read()

AST = None
ST = None
code_ok = None

print("---- PLY - Start")
try:
    AST = lex_parse(text_input)
    print("---- PLY - Done")
except Exception as e:
    print("---- PLY - Error")
    print(traceback.format_exc())
    exit(0)

# print(lex_error, syntax_error)
#print(parser.errok())
# if parser.errors:
    # exit(0)

parse_tree = AST.create_tree()
# parse_tree.pretty_print()
parse_tree.draw()

print("---- Symbol Table - Start")
try:
    ST = SymbolTable()
    ST.fill(AST)
    print("---- Symbol Table - Done")
except AssertionError as e:
    print("---- Symbol Table - Error")
    print(e)
    exit(0)
except Exception as e:
    print("---- Symbol Table - Error")
    print(traceback.format_exc())
    exit(0)


print("---- Checking - Start")
try:
    code_ok = AST.check(ST)
except AssertionError as e:
    # print(traceback.format_exc())
    print("---- Checking - Error")
    print(e)
    exit(0)
except Exception as e:
    print("---- Checking - Error")
    print(traceback.format_exc())
    exit(0)


if code_ok:
    print("---- Checking - Done")
else:
    print("---- Checking - Code contains errors")
    exit(0)


print("---- Code Generator - Start")
try:
    AST.generate(ST)
    codegen.create_ir()
    codegen.save_ir("llc/my_output.ll")
    print("---- Code Generator - Done")
except AssertionError as e:
    # print(traceback.format_exc())
    print("---- Code Generator - Error")
    print(e)
    print(traceback.format_exc())
    exit(0)
except Exception as e:
    print("---- Code Generator - Error")
    print(traceback.format_exc())
    exit(0)
