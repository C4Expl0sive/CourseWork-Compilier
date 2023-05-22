import ply.lex as lex
from ply.yacc import yacc
from ast import *
from codegen import CodeGen

codegen = CodeGen()
module = codegen.module
printf = codegen.printf
puts = codegen.puts
scanf = codegen.scanf

"""
def find_tok_linepos(token):
    last_cr = token.lexer.lexdata.rfind('\n', 0, token.lexpos)
    if last_cr < 0:
        last_cr = -1
    line = token.lexer.lexdata[last_cr+1:]
    col = (token.lexpos - last_cr)
    return token.lineno, col
"""

lex_error = False
syntax_error = False

tokens = (
    'PRINT', 'LPAREN', 'RPAREN', 'SEMI', 'COMMA', 'ASSIGN', 'COLON',
    'SUM', 'SUB', 'MUL', 'DIV', 'COMPARISON', 'NOT', 'AND', 'OR',
    'FLOAT', 'INT', 'BEGIN', 'END', 'VAR', 'TYPE', 'IF', 'DO', 'ELSE',
    'WHILE', 'BREAK', 'CONTINUE', 'DOT', 'FUNC', 'RETURN', 'READ',
    'IDENTIFIER', 'FUNC_NAME', 'LITERAL'
)


# Print
def t_PRINT(t):
    r'Print'
    return t


# Parenthesis
t_LPAREN = r'\('
t_RPAREN = r'\)'

# Semi Colon
t_SEMI = r'\;'
t_COMMA = r'\,'
t_ASSIGN = r'\:='
t_COLON = r'\:'

# Operators
t_SUM = r'\+'
t_SUB = r'\-'
t_MUL = r'\*'
t_DIV = r'\/'

t_COMPARISON = r'>|<|='


# Logic
def t_NOT(t):
    r'not'
    return t


def t_AND(t):
    r'and'
    return t


def t_OR(t):
    r'or'
    return t


# Number
t_FLOAT = r'\d+\.\d+'
t_INT = r'\d+'


# Begin End
# t_BEGIN = r'Begin'
def t_END(t):
    r'End'
    return t


def t_VAR(t):
    r'Var'
    return t


def t_TYPE(t):
    r'int|float'
    return t


def t_IF(t):
    r'If'
    return t


def t_DO(t):
    r'Do'
    return t


def t_ELSE(t):
    r'Else'
    return t


def t_WHILE(t):
    r'While'
    return t


def t_BREAK(t):
    r'Break'
    return t


def t_CONTINUE(t):
    r'Continue'
    return t


t_DOT = r'\.'


def t_FUNC(t):
    r'Func'
    return t


def t_RETURN(t):
    r'Return'
    return t


def t_READ(t):
    r'Read'
    return t


def t_BEGIN(t):
    r'Begin'
    return t


# Function Name
def t_FUNC_NAME(t):
    r'[A-Z][A-z0-9_]*'
    return t


# Identifier
def t_IDENTIFIER(t):
    r'[a-z][a-z0-9_]*'
    return t


# Literal
def t_LITERAL(t):
    r'\'[^\']*\''
    return t


def t_NEWLINE(t):
    r'\n+'
    t.lexer.lineno += len(t.value)


# Ignore spaces
t_ignore = ' \t'
t_ignore_COMMENT = r'\#.*\n'


# Error handling
def t_error(t):
    # print(t)
    # print(type(t))
    # print("Illegal character '%s'" % t.value[0])
    # lineno, lexpos = find_tok_linepos(t)
    global lex_error
    lex_error = True
    line_start = lexer.lexdata.rfind('\n', 0, t.lexpos) + 1
    col = (t.lexpos - line_start) + 1
    print(f"---- Lexer Error: Incorrect character {t.value[0]} at position {t.lineno}:{col}")
    t.lexer.skip(1)


precedence = (
    ('left', 'SUM', 'SUB', 'OR'),
    ('left', 'MUL', 'DIV', 'AND'),
    ('left', 'NOT'),
)


def p_program(p):
    """
    program : VAR var_list func_list block DOT
            | VAR var_list block DOT
            | func_list block DOT
            | block DOT
    """
    if len(p) == 5:
        p[0] = Program(module, p[3], p[2], None)
    elif len(p) == 6:
        p[0] = Program(module, p[4], p[2], p[3])
    elif len(p) == 4:
        p[0] = Program(module, p[2], None, p[1])
    elif len(p) == 3:
        p[0] = Program(module, p[1], None, None)
    # else:
    # print("some")


def p_func_list(p):
    """
    func_list : func
              | func func_list
    """
    if len(p) == 2:
        p[0] = FuncList(p[1])
    else:
        p[0] = FuncList(p[1], p[2])


def p_func(p):
    """
    func : FUNC FUNC_NAME LPAREN var_list RPAREN COLON TYPE SEMI VAR var_list block SEMI
         | FUNC FUNC_NAME LPAREN RPAREN COLON TYPE SEMI VAR var_list block SEMI
         | FUNC FUNC_NAME LPAREN RPAREN COLON TYPE SEMI block SEMI
    """
    if len(p) == 13:
        p[0] = Func(module, p[2], p[7], p[11], p[4], p[10])
    elif len(p) == 12:
        p[0] = Func(module, p[2], p[6], p[10], None, p[9])
    else:
        p[0] = Func(module, p[2], p[6], p[8], None, None)


def p_func_args_novar(p):
    """
    func : FUNC FUNC_NAME LPAREN var_list RPAREN COLON TYPE SEMI block SEMI
    """
    p[0] = Func(module, p[2], p[7], p[9], p[4], None)


def p_var_list(p):
    """
    var_list : IDENTIFIER COLON TYPE SEMI
    | IDENTIFIER COLON TYPE SEMI var_list
    """
    if len(p) == 5:
        p[0] = VarList(module, p[1], p[3])
    else:
        p[0] = VarList(module, p[1], p[3], p[5])


def p_block(p):
    """
    block : BEGIN statements_list END
    | statement
    """
    if len(p) == 4:
        p[0] = Block(p[2])
    elif len(p) == 2:
        p[0] = Block(StatementsList(p[1]))
    # else:
    # print("some")


def p_statements_list(p):
    """
    statements_list : statement SEMI
    | statement SEMI statements_list
    """
    if len(p) == 3:
        p[0] = StatementsList(p[1])
    else:
        p[0] = StatementsList(p[1], p[3])


def p_statements_list_err(p):
    """
    statements_list : statement
    | statement statements_list
    """
    global syntax_error
    syntax_error = True

    # print(lexer.lexdata)
    # print(p.lexspan(1))
    line_start = lexer.lexdata.rfind('\n', 0, p.lexspan(1)[1]) + 1
    col = (p.lexspan(1)[1] - line_start) + 1

    print(f'---- Syntax Error: Missing SEMICOLON after position {p.lineno(1)}:{col}')
    if len(p) == 2:
        p[0] = StatementsList(p[1])
    else:
        p[0] = StatementsList(p[1], p[2])


def p_statement(p):
    """
    statement : assignment_statement
    | if_statement
    | while_statement
    | print_statement
    | read_statement
    | func_call
    | BREAK
    | CONTINUE
    | RETURN expression
    """

    if len(p) == 3:
        p[0] = Return(printf, puts, p[2])

    elif isinstance(p[1], str):
        if p[1] == 'Break':
            p[0] = Break()

        elif p[1] == 'Continue':
            p[0] = Continue()

    else:
        p[0] = p[1]


def p_read_statement(p):
    """
    read_statement : READ variable
    """
    p[0] = ReadStatement(p[2], scanf)


def p_assignment_statement(p):
    """
    assignment_statement : variable ASSIGN expression
    """
    p[0] = AssignmentStatement(printf, puts, p[1], p[3])


def p_if_statement(p):
    """
    if_statement : IF bool DO block ELSE block
    | IF bool DO block
    """
    if len(p) == 5:
        p[0] = IfStatement(printf, puts, p[2], p[4])
    else:
        p[0] = IfStatement(printf, puts, p[2], p[4], p[6])


def p_while_statement(p):
    """
    while_statement : WHILE bool DO block
    """
    p[0] = WhileStatement(printf, puts, p[2], p[4])


def p_print_statement(p):
    """
    print_statement : PRINT expression
    | PRINT LITERAL
    """
    # print(p[2])
    p[0] = PrintStatement(printf, puts, p[2])


def p_expression_0(p):
    """
    expression : expression SUM expression
    | expression SUB expression
    """
    left = p[1]
    right = p[3]
    operator = p[2]
    # print(operator)
    if operator == '+':
        p[0] = Sum(left, right)
    elif operator == '-':
        p[0] = Sub(left, right)


def p_expression_1(p):
    """
    expression : expression MUL expression
    |  expression DIV expression
    """
    left = p[1]
    right = p[3]
    operator = p[2]
    if operator == '*':
        p[0] = Mul(left, right)
    elif operator == '/':
        p[0] = Div(left, right)


def p_expression_2(p):
    """
    expression : LPAREN expression RPAREN
    """
    p[0] = p[2]


def p_exp_variable(p):
    """
    expression : variable
    """
    p[0] = p[1]


def p_exp_func_call(p):
    """
    expression : func_call
    """
    p[0] = p[1]


def p_func_call(p):
    """
    func_call : FUNC_NAME LPAREN param_list RPAREN
    |  FUNC_NAME LPAREN RPAREN
    """
    if len(p) == 5:
        p[0] = FuncCall(p[1], p[3])
    else:
        p[0] = FuncCall(p[1])


def p_param_list(p):
    """
    param_list : expression
    | expression COMMA param_list
    """
    if len(p) == 2:
        p[0] = ParamList(p[1])
    else:
        p[0] = ParamList(p[1], p[3])


def p_variable(p):
    """
    variable : IDENTIFIER
    """
    p[0] = Variable(module, p[1])


def p_int(p):
    """
    expression : INT
    """
    # print(p[1])
    p[0] = Int(p[1])


def p_float(p):
    """
    expression : FLOAT
    """
    p[0] = Float(p[1])


def p_negative(p):
    """
    expression : SUB expression
    """
    # print(p[2])
    p[0] = Negative(p[2])


def p_bool2(p):
    """
    bool : bool AND bool
    | bool OR bool
    """
    left = p[1]
    right = p[3]
    operator = p[2]
    if operator == 'and':
        p[0] = And(left, right)
    elif operator == 'or':
        p[0] = Or(left, right)


def p_bool1(p):
    """
    bool : NOT bool
    """
    p[0] = Not(p[2])


def p_bool0(p):
    """
    bool : comparison
    """
    p[0] = Bool(p[1])


def p_bool3(p):
    """
    bool : LPAREN bool RPAREN
    """
    p[0] = p[2]


def p_comparison(p):
    """
    comparison : expression COMPARISON expression
    """
    left = p[1]
    right = p[3]
    operator = p[2]
    if operator == '>':
        p[0] = More(left, right)
    elif operator == '<':
        p[0] = Less(left, right)
    else:
        p[0] = Equal(left, right)


def p_error(p):
    # lineno, lexpos = find_tok_linepos(p)
    # print(00)
    global syntax_error
    if syntax_error and (p.value == ';' or p.value == 'End'):
        parser.errok()
        # tok = parser.token()
        # parser.restart()
        return
    syntax_error = True

    if p:
        line_start = lexer.lexdata.rfind('\n', 0, p.lexpos) + 1
        col = (p.lexpos - line_start) + 1
        print(f'---- Syntax Error: Unexpected token {p.value} at position {p.lineno}:{col}')
        '''
        stack_state_str = ' '.join([symbol.type for symbol in parser.symstack][1:])
        print('Syntax error in input! Parser State:{} {} . {}'
              .format(parser.state,
                      stack_state_str,
                      p))
        '''

        while True:
            tok = parser.token()  # Get the next token
            if not tok or tok.type == 'SEMI': break
        parser.restart()
    else:
        print(f'---- Syntax Error: Unexpected EOF')


lexer = lex.lex()
parser = yacc()


def lex_parse(text_input):
    global syntax_error
    '''
    begin_end_counter = 0
    cur_begin = None
    lexer.input(text_input)
    while True:
        tok = lexer.token()
        # print(tok)
        if not tok:
            break
        if tok.type == 'BEGIN':
            begin_end_counter += 1
            cur_begin = tok
        elif tok.type == 'END':
            begin_end_counter -= 1
            if begin_end_counter < 0:
                line_start = lexer.lexdata.rfind('\n', 0, tok.lexpos) + 1
                col = (tok.lexpos - line_start) + 1
                print(f'---- Syntax Error: "End" without "Begin" at position {tok.lineno}:{col}')
                syntax_error = True

    if begin_end_counter > 0:
        line_start = lexer.lexdata.rfind('\n', 0, cur_begin.lexpos) + 1
        col = (cur_begin.lexpos - line_start) + 1
        print(f'---- Syntax Error: "Begin" without "End" at position {cur_begin.lineno}:{col}')
        syntax_error = True

    lexer.lineno = 1
    '''
    AST = parser.parse(text_input, tracking=True)
    if lex_error or syntax_error:
        exit(0)
    else:
        return AST
