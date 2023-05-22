from nltk.tree import Tree
from rply.token import Token
from llvmlite import ir

# int_fmt_arg = None
# float_fmt_arg = None

# int_fmt = None
# float_fmt = None


def bitcast_fstrings(builder, int_fmt, float_fmt, read_fmt):
    voidptr_ty = ir.IntType(32).as_pointer()
    # global int_fmt
    int_fmt_arg = builder.bitcast(int_fmt, voidptr_ty)
    # global float_fmt
    float_fmt_arg = builder.bitcast(float_fmt, voidptr_ty)
    # global read_fmt
    read_fmt_arg = builder.bitcast(read_fmt, voidptr_ty)

    return int_fmt_arg, float_fmt_arg, read_fmt_arg


class Int():
    def __init__(self, value):
        # self.builder = builder
        # self.module = module
        # self.printf = printf
        self.value = value

    def to_string(self):
        print("int", self.value)
        # print(self.value)

    def create_tree(self):
        parse_tree = Tree("Int", [self.value])
        return parse_tree

    def check(self, st, scope):
        return True

    def is_float(self, st, scope):
        return False

    def generate(self, st, scope, builder):
        i = ir.Constant(ir.IntType(32), int(self.value))
        return i

    def is_const(self, st, scope):
        return True

    def eval(self, st, scope):
        return int(self.value)


class Float():
    def __init__(self, value):
        # self.builder = builder
        # self.module = module
        # self.printf = printf
        self.value = value

    def to_string(self):
        print("float", self.value)

    def create_tree(self):
        parse_tree = Tree("Float", [self.value])
        return parse_tree

    def check(self, st, scope):
        return True

    def is_float(self, st, scope):
        return True

    def generate(self, st, scope, builder):
        # print(float(self.value))
        i = ir.Constant(ir.DoubleType(), float(self.value))
        return i

    def is_const(self, st, scope):
        return True

    def eval(self, st, scope):
        return float(self.value)


class BinaryOp():
    def __init__(self, left, right):
        # self.builder = builder
        # self.module = module
        # self.printf = printf
        self.left = left
        self.right = right

    def is_const(self, st, scope):
        return self.left.is_const(st, scope) and self.right.is_const(st, scope)


class Negative():
    def __init__(self, expr):
        # self.builder = builder
        # self.module = module
        # self.printf = printf
        self.expr = expr

    def to_string(self):
        print("Negative")
        self.expr.to_string()

    def create_tree(self):
        parse_tree = Tree("Negative", [self.expr.create_tree()])
        return parse_tree

    def check(self, st, scope):
        return self.expr.check(st, scope)

    def is_float(self, st, scope):
        return self.expr.is_float(st, scope)

    def generate(self, st, scope, builder):
        if not self.is_float(st, scope):
            i = builder.neg(self.expr.generate(st, scope, builder))
        else:
            i_expr = builder.sitofp(self.expr.generate(st, scope, builder), ir.DoubleType())
            i = builder.fneg(i_expr)

        return i

    def is_const(self, st, scope):
        return self.expr.is_const(st, scope)

    def eval(self, st, scope):
        return -self.expr.eval(st, scope)


class Sum(BinaryOp):
    def to_string(self):
        print("Sum")
        self.left.to_string()
        self.right.to_string()

    def create_tree(self):
        parse_tree = Tree("Sum", [self.left.create_tree(), self.right.create_tree()])
        return parse_tree

    def check(self, st, scope):
        return self.left.check(st, scope) and self.right.check(st, scope)

    def is_float(self, st, scope):
        return self.left.is_float(st, scope) or self.right.is_float(st, scope)

    def generate(self, st, scope, builder):
        if not self.is_float(st, scope):
            i = builder.add(self.left.generate(st, scope, builder), self.right.generate(st, scope, builder))
        else:
            if not self.left.is_float(st, scope):
                tmp = self.left.generate(st, scope, builder)
                # print(tmp)
                i_left = builder.sitofp(tmp, ir.DoubleType())
                i_right = self.right.generate(st, scope, builder)
            else:
                i_left = self.left.generate(st, scope, builder)
                i_right = builder.sitofp(self.right.generate(st, scope, builder), ir.DoubleType())

            i = builder.fadd(i_left, i_right)

        return i

    def eval(self, st, scope):
        return self.left.eval(st, scope) + self.right.eval(st, scope)


class Sub(BinaryOp):
    def to_string(self):
        print("Sub")
        self.left.to_string()
        self.right.to_string()

    def create_tree(self):
        parse_tree = Tree("Sub", [self.left.create_tree(), self.right.create_tree()])
        return parse_tree

    def check(self, st, scope):
        return self.left.check(st, scope) and self.right.check(st, scope)

    def is_float(self, st, scope):
        return self.left.is_float(st, scope) or self.right.is_float(st, scope)

    def generate(self, st, scope, builder):
        if not self.is_float(st, scope):
            i = builder.sub(self.left.generate(st, scope, builder), self.right.generate(st, scope, builder))
        else:
            if not self.left.is_float(st, scope):
                tmp = self.left.generate(st, scope, builder)
                # print(tmp)
                i_left = builder.sitofp(tmp, ir.DoubleType())
                i_right = self.right.generate(st, scope, builder)
            else:
                i_left = self.left.generate(st, scope, builder)
                i_right = builder.sitofp(self.right.generate(st, scope, builder), ir.DoubleType())

            i = builder.fsub(i_left, i_right)

        return i

    def eval(self, st, scope):
        return self.left.eval(st, scope) - self.right.eval(st, scope)


class Mul(BinaryOp):
    def to_string(self):
        print("Mul")
        self.left.to_string()
        self.right.to_string()

    def create_tree(self):
        parse_tree = Tree("Mul", [self.left.create_tree(), self.right.create_tree()])
        return parse_tree

    def check(self, st, scope):
        return self.left.check(st, scope) and self.right.check(st, scope)

    def is_float(self, st, scope):
        return self.left.is_float(st, scope) or self.right.is_float(st, scope)

    def generate(self, st, scope, builder):
        if not self.is_float(st, scope):
            i = builder.mul(self.left.generate(st, scope, builder), self.right.generate(st, scope, builder))
        else:
            if not self.left.is_float(st, scope):
                tmp = self.left.generate(st, scope, builder)
                # print(tmp)
                i_left = builder.sitofp(tmp, ir.DoubleType())
                i_right = self.right.generate(st, scope, builder)
            else:
                i_left = self.left.generate(st, scope, builder)
                i_right = builder.sitofp(self.right.generate(st, scope, builder), ir.DoubleType())

            i = builder.fmul(i_left, i_right)

        return i

    def eval(self, st, scope):
        return self.left.eval(st, scope) * self.right.eval(st, scope)


class Div(BinaryOp):
    def to_string(self):
        print("Div")
        self.left.to_string()
        self.right.to_string()

    def create_tree(self):
        parse_tree = Tree("Div", [self.left.create_tree(), self.right.create_tree()])
        return parse_tree

    def check(self, st, scope):
        return self.left.check(st, scope) and self.right.check(st, scope)

    def is_float(self, st, scope):
        return self.left.is_float(st, scope) or self.right.is_float(st, scope)

    def generate(self, st, scope, builder):
        if not self.is_float(st, scope):
            i = builder.sdiv(self.left.generate(st, scope, builder), self.right.generate(st, scope, builder))
        else:
            i_left = self.left.generate(st, scope, builder)
            i_right = self.right.generate(st, scope, builder)
            if not self.left.is_float(st, scope):
                i_left = builder.sitofp(i_left, ir.DoubleType())
            if not self.right.is_float(st, scope):
                i_right = builder.sitofp(i_right, ir.DoubleType())

            i = builder.fdiv(i_left, i_right)

        return i

    def eval(self, st, scope):
        return self.left.eval(st, scope) / self.right.eval(st, scope)


class And(BinaryOp):
    def to_string(self):
        print("And")
        self.left.to_string()
        self.right.to_string()

    def create_tree(self):
        parse_tree = Tree("And", [self.left.create_tree(), self.right.create_tree()])
        return parse_tree

    def check(self, st, scope):
        return self.left.check(st, scope) and self.right.check(st, scope)

    '''
    def is_float(self, st, scope):
        raise Exception('is float bool')
        # return True
    '''

    def generate(self, st, scope, builder):
        i = builder.and_(self.left.generate(st, scope, builder), self.right.generate(st, scope, builder))
        return i

    def eval(self, st, scope):
        return self.left.eval(st, scope) and self.right.eval(st, scope)


class Or(BinaryOp):
    def to_string(self):
        print("Or")
        self.left.to_string()
        self.right.to_string()

    def create_tree(self):
        parse_tree = Tree("Or", [self.left.create_tree(), self.right.create_tree()])
        return parse_tree

    def check(self, st, scope):
        return self.left.check(st, scope) and self.right.check(st, scope)

    '''
    def is_float(self, st, scope):
        raise Exception('is float bool')
        # return True
    '''

    def generate(self, st, scope, builder):
        i = builder.or_(self.left.generate(st, scope, builder), self.right.generate(st, scope, builder))
        return i

    def eval(self, st, scope):
        return self.left.eval(st, scope) or self.right.eval(st, scope)


class Not():
    def __init__(self, bool):
        # self.builder = builder
        # self.module = module
        # self.printf = printf
        self.bool = bool

    def to_string(self):
        print("Not")
        self.bool.to_string()

    def create_tree(self):
        parse_tree = Tree("Not", [self.bool.create_tree()])
        return parse_tree

    def check(self, st, scope):
        return self.bool.check(st, scope)

    '''
    def is_float(self, st, scope):
        raise Exception('is float bool')
        # return True
    '''

    def generate(self, st, scope, builder):
        i = builder.not_(self.bool.generate(st, scope, builder))
        return i

    def is_const(self, st, scope):
        return self.bool.is_const(st, scope)


class More(BinaryOp):
    def to_string(self):
        print("More")
        self.left.to_string()
        self.right.to_string()

    def create_tree(self):
        parse_tree = Tree("More", [self.left.create_tree(), self.right.create_tree()])
        return parse_tree

    def check(self, st, scope):
        return self.left.check(st, scope) and self.right.check(st, scope)

    '''
    def is_float(self, st, scope):
        raise Exception('is float bool')
        # return True
    '''

    def generate(self, st, scope, builder):
        # fix to compare float and int
        if not (self.left.is_float(st, scope) or self.right.is_float(st, scope)):
            i = builder.icmp_signed('>', self.left.generate(st, scope, builder), self.right.generate(st, scope, builder))
        else:
            if not self.left.is_float(st, scope):
                tmp = self.left.generate(st, scope, builder)
                i_left = builder.sitofp(tmp, ir.DoubleType())
                i = builder.fcmp_ordered('>', i_left, self.right.generate(st, scope, builder))
            else:
                tmp = self.right.generate(st, scope, builder)
                i_right = builder.sitofp(tmp, ir.DoubleType())
                i = builder.fcmp_ordered('>', self.left.generate(st, scope, builder), i_right)

        return i

    def eval(self, st, scope):
        return self.left.eval(st, scope) > self.right.eval(st, scope)


class Less(BinaryOp):
    def to_string(self):
        print("Less")
        self.left.to_string()
        self.right.to_string()

    def create_tree(self):
        parse_tree = Tree("Less", [self.left.create_tree(), self.right.create_tree()])
        return parse_tree

    def check(self, st, scope):
        return self.left.check(st, scope) and self.right.check(st, scope)

    '''
    def is_float(self, st, scope):
        raise Exception('is float bool')
        # return True
    '''

    def generate(self, st, scope, builder):
        # fix to compare float and int
        if not (self.left.is_float(st, scope) or self.right.is_float(st, scope)):
            i = builder.icmp_signed('<', self.left.generate(st, scope, builder), self.right.generate(st, scope, builder))
        else:
            if not self.left.is_float(st, scope):
                tmp = self.left.generate(st, scope, builder)
                i_left = builder.sitofp(tmp, ir.DoubleType())
                i = builder.fcmp_ordered('<', i_left, self.right.generate(st, scope, builder))
            else:
                tmp = self.right.generate(st, scope, builder)
                i_right = builder.sitofp(tmp, ir.DoubleType())
                i = builder.fcmp_ordered('<', self.left.generate(st, scope, builder), i_right)

        return i

    def eval(self, st, scope):
        return self.left.eval(st, scope) < self.right.eval(st, scope)


class Equal(BinaryOp):
    def to_string(self):
        print("Equal")
        self.left.to_string()
        self.right.to_string()

    def create_tree(self):
        parse_tree = Tree("Equal", [self.left.create_tree(), self.right.create_tree()])
        return parse_tree

    def check(self, st, scope):
        return self.left.check(st, scope) and self.right.check(st, scope)

    '''
    def is_float(self, st, scope):
        raise Exception('is float bool')
        # return True
    '''

    def generate(self, st, scope, builder):
        # fix to compare float and int
        if not (self.left.is_float(st, scope) or self.right.is_float(st, scope)):
            i = builder.icmp_signed('==', self.left.generate(st, scope, builder), self.right.generate(st, scope, builder))
        else:
            if not self.left.is_float(st, scope):
                tmp = self.left.generate(st, scope, builder)
                i_left = builder.sitofp(tmp, ir.DoubleType())
                i = builder.fcmp_ordered('==', i_left, self.right.generate(st, scope, builder))
            else:
                tmp = self.right.generate(st, scope, builder)
                i_right = builder.sitofp(tmp, ir.DoubleType())
                i = builder.fcmp_ordered('==', self.left.generate(st, scope, builder), i_right)

        return i

    def eval(self, st, scope):
        return self.left.eval(st, scope) == self.right.eval(st, scope)


class Bool():
    def __init__(self, bool):
        # self.builder = builder
        # self.module = module
        # self.printf = printf
        self.bool = bool
        # print("Bool")
        # self.to_string()

    def to_string(self):
        print("Bool")
        self.bool.to_string()

    def create_tree(self):
        parse_tree = Tree("Bool", [self.bool.create_tree()])
        return parse_tree

    def check(self, st, scope):
        return self.bool.check(st, scope)

    '''
    def is_float(self, st, scope):
        raise Exception('is float bool')
        # return True
    '''

    def generate(self, st, scope, builder):
        return self.bool.generate(st, scope, builder)

    def is_const(self, st, scope):
        return self.bool.is_const(st, scope)

    def eval(self, st, scope):
        return self.bool.eval(st, scope)

class Statement():
    def __init__(self, printf, puts, block1, block2=None, block3=None):
        # self.builder = builder
        # self.module = module
        self.printf = printf
        self.puts = puts
        # self.base_func = base_func
        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.not_used = False
        # print(block1)

    def to_string(self):
        pass

    def create_tree(self):
        parse_tree = Tree("Statement", [self.block1])
        return parse_tree


class AssignmentStatement(Statement):
    def to_string(self):
        print("AssignmentStatement")
        print(self.block1)
        self.block2.to_string()

    def create_tree(self):
        parse_tree = Tree("AssignmentStatement", [self.block1.create_tree(), self.block2.create_tree()])
        return parse_tree

    def check(self, st, scope):
        # var_name = self.block1.variable
        # if not st.var_in_scope(var_name, scope):
        if not self.block1.check(st, scope, self):
            return False

        if not self.block2.check(st, scope):
            return False

        # var = st.get_var_by_name(var_name, scope)
        # if var.dtype == 'int' and self.block2.is_float(st, scope):
        if (not self.block1.is_float(st, scope)) and self.block2.is_float(st, scope):
            # assert False, f"CompileError: Can't assign float value to int variable {self.block1.variable}"
            # return False
            return True

        return True
        # return self.left.check(st, scope) and self.right.check(st, scope)

    '''
    def is_float(self, st, scope):
        raise Exception('is float st')
        # return True
    '''

    def generate(self, st, scope, builder):
        # self.block1.declare(st, scope)
        # self.block1.assign(self.block2.generate(st, scope, builder), st, scope)
        if self.not_used:
            return
        if self.block2.is_const(st, scope):
            right_val = self.block2.eval(st, scope)
            # print(right_val)
            # self.block1.set_const(right_val)
            if not self.block2.is_float(st, scope):
                right = ir.Constant(ir.IntType(32), right_val)
            else:
                right = ir.Constant(ir.DoubleType(), right_val)
        else:
            right = self.block2.generate(st, scope, builder)

        left = self.block1.assign(st, scope, builder)

        if self.block1.is_float(st, scope):
            if not self.block2.is_float(st, scope):
                right = builder.sitofp(right, ir.DoubleType())
        else:
            if self.block2.is_float(st, scope):
                right = builder.fptosi(right, ir.IntType(32))

        i = builder.store(right, left)


        # self.block1.reload(st, scope)
        ## return i


class IfStatement(Statement):
    def to_string(self):
        print("IfStatement")
        self.block1.to_string()
        self.block2.to_string()
        if self.block3 is not None:
            self.block3.to_string()

    def create_tree(self):
        if self.block3 is not None:
            parse_tree = Tree("IfStatement",
                              [self.block1.create_tree(), self.block2.create_tree(), self.block3.create_tree()])
        else:
            parse_tree = Tree("IfStatement", [self.block1.create_tree(), self.block2.create_tree()])
        return parse_tree

    def check(self, st, scope, in_cycle=False):
        # print(in_cycle)
        if not self.block1.check(st, scope):
            return False

        if not self.block2.check(st, scope, in_cycle):
            return False

        if self.block3 is not None:
            if not self.block3.check(st, scope, in_cycle):
                return False

        return True

    '''
    def is_float(self, st, scope):
        raise Exception('is float st')
        # return True
    '''

    def generate(self, st, scope, builder, base_func, int_fmt_arg, float_fmt_arg, read_fmt_arg, c_head_block, c_end_block, prev_end_block=None):
        # print(self.block1.eval(st, scope))
        if self.block1.is_const(st, scope):
            if self.block1.eval(st, scope):
                self.block2.generate(st, scope, builder, base_func, int_fmt_arg, float_fmt_arg, read_fmt_arg, c_head_block, c_end_block)
            else:
                if self.block3:
                    self.block3.generate(st, scope, builder, base_func, int_fmt_arg, float_fmt_arg, read_fmt_arg, c_head_block, c_end_block)
            return

        condition = self.block1.generate(st, scope, builder)
        then_block = base_func.append_basic_block()
        if self.block3 is None:
            # if prev_end_block is None:
                # end_block = base_func.append_basic_block()
            # else:
                # end_block = prev_end_block
            end_block = base_func.append_basic_block()
            builder.cbranch(condition, then_block, end_block)

            builder.position_at_end(then_block)
            self.block2.generate(st, scope, builder, base_func, int_fmt_arg, float_fmt_arg, read_fmt_arg, c_head_block, c_end_block, end_block, if_block=True)

            if not then_block.is_terminated:
                builder.branch(end_block)

            builder.position_at_end(end_block)
            # if prev_end_block is not None:
                # builder.branch(prev_end_block)

        else:
            end_block = None
            else_block = base_func.append_basic_block()
            builder.cbranch(condition, then_block, else_block)

            builder.position_at_end(then_block)
            self.block2.generate(st, scope, builder, base_func, int_fmt_arg, float_fmt_arg, read_fmt_arg, c_head_block, c_end_block)
            if not then_block.is_terminated:
                end_block = base_func.append_basic_block()
                # print(end_block)
                self.block2.terminate_if(builder, end_block)
                # if prev_end_block is None:
                    # end_block = base_func.append_basic_block()
                # else:
                    # end_block = prev_end_block
                # builder.branch(end_block)
                # if prev_end_block is not None:
                    # builder.branch(prev_end_block)

            builder.position_at_end(else_block)

            # print(end_block)

            if end_block is None:
                self.block3.generate(st, scope, builder, base_func, int_fmt_arg, float_fmt_arg, read_fmt_arg, c_head_block, c_end_block)
            else:
                self.block3.generate(st, scope, builder, base_func, int_fmt_arg, float_fmt_arg, read_fmt_arg, c_head_block, c_end_block, end_block, if_block=True)

            if not else_block.is_terminated:
                assert end_block is not None, "CompileError: Missing Return statament"

                builder.branch(end_block)
                builder.position_at_end(end_block)
                # if prev_end_block is not None:
                    # builder.branch(prev_end_block)


class WhileStatement(Statement):
    def to_string(self):
        print("WhileStatement")
        self.block1.to_string()
        self.block2.to_string()

    def create_tree(self):
        parse_tree = Tree("WhileStatement", [self.block1.create_tree(), self.block2.create_tree()])
        return parse_tree

    def check(self, st, scope):
        if not self.block1.check(st, scope):
            return False

        if not self.block2.check(st, scope, True):
            return False

        return True

    '''
    def is_float(self, st, scope):
        raise Exception('is float st')
        # return True
    '''

    def generate(self, st, scope, builder, base_func, int_fmt_arg, float_fmt_arg, read_fmt_arg):
        if self.block1.is_const(st, scope):
            if not self.block1.eval(st, scope):
                return
        # fix according to picture
        condition_block = base_func.append_basic_block()
        builder.branch(condition_block)
        builder.position_at_end(condition_block)
        condition = self.block1.generate(st, scope, builder)

        do_block = base_func.append_basic_block()
        end_block = base_func.append_basic_block()
        # print(condition_block)
        # print(do_block)
        # print(end_block)

        builder.cbranch(condition, do_block, end_block)

        builder.position_at_end(do_block)
        self.block2.generate(st, scope, builder, base_func, int_fmt_arg, float_fmt_arg, read_fmt_arg, c_head_block=condition_block, c_end_block=end_block)
        builder.branch(condition_block)

        builder.position_at_end(end_block)


class PrintStatement(Statement):
    def to_string(self):
        print("PrintStatement")
        self.block1.to_string()

    def create_tree(self):
        if isinstance(self.block1, str):
            parse_tree = Tree('PrintStatement', [Tree("Literal", [self.block1])])
        else:
            parse_tree = Tree("PrintStatement", [self.block1.create_tree()])
        return parse_tree

    def check(self, st, scope):
        if isinstance(self.block1, str):
            return True

        # print
        if not self.block1.check(st, scope):
            return False

        return True

    '''
    def is_float(self, st, scope):
        raise Exception('is float st')
        # return True
    '''

    def generate(self, st, scope, builder, int_fmt_arg, float_fmt_arg):
        # print(self.block1.g_var)
        if not isinstance(self.block1, str):
            to_print = self.block1.generate(st, scope, builder)

            # Call Print Function
            if not self.block1.is_float(st, scope):
                builder.call(self.printf, [int_fmt_arg, to_print])
            else:
                builder.call(self.printf, [float_fmt_arg, to_print])

        else:
            print_str = self.block1[1:-1]
            # ir_str = ir.Constant(ir.ArrayType(ir.IntType(8), len(print_str) + 1), bytearray(print_str + "\0", 'utf8'))
            # ir_format = builder.bitcast(ir_str, ir.PointerType(ir.IntType(8)))
            # builder.call(self.puts, [ir_format])

            str_ptr = builder.alloca(ir.ArrayType(ir.IntType(8), len(print_str) + 1))
            string_constant = ir.Constant(ir.ArrayType(ir.IntType(8), len(print_str) + 1),
                                          bytearray(print_str.encode("utf8") + b"\00"))
            builder.store(string_constant, str_ptr)
            elem_ptr = builder.gep(str_ptr, [ir.IntType(32)(0), ir.IntType(32)(0)])
            builder.call(self.puts, [elem_ptr])


class Block():
    def __init__(self, statements_list):
        # self.builder = builder
        # self.module = module
        # self.printf = printf
        self.statements_list = statements_list
        # print("Block")

    def to_string(self):
        # print("Block")
        self.statements_list.to_string()

    def create_tree(self):
        parse_tree = Tree("Block", [self.statements_list.create_tree()])
        return parse_tree

    def check(self, st, scope, in_cycle=False):
        # print(in_cycle)
        if not self.statements_list.check(st, scope, in_cycle):
            return False

        return True

    '''
    def is_float(self, st, scope):
        raise Exception('is float block')
    '''

    def generate(self, st, scope, builder, base_func, int_fmt_arg, float_fmt_arg, read_fmt_arg, c_head_block=None, c_end_block=None, if_end_block=None, if_block=False, ret_type=None):
        # print(if_end_block)
        self.statements_list.generate(st, scope, builder, base_func, int_fmt_arg, float_fmt_arg,  read_fmt_arg, c_head_block, c_end_block, if_end_block)
        # print(ir_block.is_terminated)
        # print(ir_block)

        # print(builder.block.is_terminated)
        # print(builder.block)

        if if_block:
            # print(builder.block)
            # print("ok")
            # print(if_end_block)
            self.terminate_if(builder, if_end_block)

    def terminate_if(self, builder, if_end_block):
        if not builder.block.is_terminated:
            # print("ok")
            # print(if_end_block)
            builder.branch(if_end_block)
            builder.position_at_end(if_end_block)

    # def contains_return(self):
        # return


class StatementsList():
    def __init__(self, statement, statements_list=None):
        # self.builder = builder
        # self.module = module
        # self.printf = printf
        self.statement = statement
        self.statements_list = statements_list

    def to_string(self):
        print("StatementsList")
        self.statement.to_string()

        if self.statements_list is not None:
            self.statements_list.to_string()

    def create_tree(self):
        if self.statements_list is not None:
            parse_tree = Tree("StatementsList", [self.statement.create_tree(), self.statements_list.create_tree()])
        else:
            # print(self.statement)
            parse_tree = Tree("StatementsList", [self.statement.create_tree()])
        return parse_tree

    def check(self, st, scope, in_cycle=False):
        # print(self.statement)

        if isinstance(self.statement, Break) or isinstance(self.statement, Continue) or isinstance(self.statement, IfStatement):
            # print("ok")
            st_ok = self.statement.check(st, scope, in_cycle)
        else:
            # print(self.statement)
            st_ok = self.statement.check(st, scope)

        if not st_ok:
            return False

        if self.statements_list is not None:
            if not self.statements_list.check(st, scope, in_cycle):
                return False

        return True

    '''
    def is_float(self, st, scope):
        raise Exception('is float st block')
    '''

    def generate(self, st, scope, builder, base_func, int_fmt_arg, float_fmt_arg, read_fmt_arg, c_head_block=None, c_end_block=None, if_end_block=None):
        if isinstance(self.statement, PrintStatement):
            self.statement.generate(st, scope, builder, int_fmt_arg, float_fmt_arg)
        elif isinstance(self.statement, WhileStatement):
            self.statement.generate(st, scope, builder, base_func, int_fmt_arg, float_fmt_arg, read_fmt_arg)
        elif isinstance(self.statement, IfStatement):
            self.statement.generate(st, scope, builder, base_func, int_fmt_arg, float_fmt_arg, read_fmt_arg, c_head_block, c_end_block, if_end_block)
        elif isinstance(self.statement, AssignmentStatement):
            self.statement.generate(st, scope, builder)
        elif isinstance(self.statement, Return):
            self.statement.generate(st, scope, builder)
        elif isinstance(self.statement, Break):
            self.statement.generate(st, scope, builder, c_end_block)
        elif isinstance(self.statement, Continue):
            self.statement.generate(st, scope, builder, c_head_block)
        elif isinstance(self.statement, ReadStatement):
            self.statement.generate(st, scope, builder, read_fmt_arg)
        elif isinstance(self.statement, FuncCall):
            self.statement.generate(st, scope, builder)

        if self.statements_list is not None:
            self.statements_list.generate(st, scope, builder, base_func, int_fmt_arg, float_fmt_arg, read_fmt_arg, c_head_block, c_end_block, if_end_block)


class VarList():
    def __init__(self, module, var, var_type, var_list=None):
        # self.builder = builder
        self.module = module
        # self.printf = printf
        self.var = var
        self.var_type = var_type
        self.var_list = var_list

    def to_string(self):
        print("VarList")
        print(self.var, self.var_type)

        if self.var_list is not None:
            self.var_list.to_string()

    def create_tree(self):
        if self.var_list is not None:
            parse_tree = Tree("VarList", [self.var + ' ' + self.var_type, self.var_list.create_tree()])
        else:
            parse_tree = Tree("VarList", [self.var + ' ' + self.var_type])
        return parse_tree

    def get_list(self):
        if self.var_list is None:
            return [[self.var, self.var_type]]
        else:
            return [[self.var, self.var_type]] + self.var_list.get_list()

    def check(self, st, scope):
        return True

    def generate(self, st, scope, builder):
        st_var = st.get_var_by_name(self.var, scope)
        st_var.init0(builder, self.module)
        if self.var_list is not None:
            self.var_list.generate(st, scope, builder)


class Program():
    def __init__(self, module, block, var_list=None, func_list=None):
        # # self.builder = builder
        self.builder = None
        self.module = module
        # self.printf = printf
        self.block = block
        self.var_list = var_list
        self.func_list = func_list
        self.base_func = None
        # print("Program")

    def to_string(self):
        print("Program")
        if self.var_list is not None:
            self.var_list.to_string()
        if self.func_list is not None:
            self.func_list.to_string()
        self.block.to_string()

    def create_tree(self):
        if self.var_list is not None and self.func_list is None:
            parse_tree = Tree("Program", [self.var_list.create_tree(),
                                          self.block.create_tree()])

        elif self.var_list is None and self.func_list is not None:
            parse_tree = Tree("Program", [self.func_list.create_tree(),
                                          self.block.create_tree()])

        elif self.var_list is not None and self.func_list is not None:
            parse_tree = Tree("Program", [self.func_list.create_tree(),
                                          self.var_list.create_tree(),
                                          self.block.create_tree()])
        else:
            parse_tree = Tree("Program", [self.block.create_tree()])

        return parse_tree

    def check(self, st):
        # print(self.block)
        if not self.block.check(st, 'global'):
            return False

        if self.var_list is not None:
            if not self.var_list.check(st, 'global'):
                return False

        if self.func_list is not None:
            if not self.func_list.check(st, 'global'):
                return False

        return True

    def generate(self, st):
        # create ir builder
        self.builder = ir.IRBuilder()

        '''
        # declare fstrings
        def declare_int_fmt_str():
            # Declare argument list
            voidptr_ty = ir.IntType(32).as_pointer()
            fmt = "%i\n\0"
            c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt)),
                                bytearray(fmt.encode("utf8")))
            global_fmt = ir.GlobalVariable(self.module, c_fmt.type, name="ifstr")
            global_fmt.linkage = 'internal'
            global_fmt.global_constant = True
            global_fmt.initializer = c_fmt
            global integer_fmt_arg
            integer_fmt_arg = self.builder.bitcast(global_fmt, voidptr_ty)

        def declare_float_fmt_str():
            # Declare argument list
            voidptr_ty = ir.IntType(32).as_pointer()
            fmt = "%.8g\n\0"
            c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(fmt)),
                                bytearray(fmt.encode("utf8")))
            global_fmt = ir.GlobalVariable(self.module, c_fmt.type, name="ffstr")
            global_fmt.linkage = 'internal'
            global_fmt.global_constant = True
            global_fmt.initializer = c_fmt
            global float_fmt_arg
            float_fmt_arg = self.builder.bitcast(global_fmt, voidptr_ty)

        declare_int_fmt_str()
        declare_float_fmt_str()
        '''

        def declare_fstrings():
            # print
            # int
            int_fmt = "%i\n\0"
            int_c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(int_fmt)),
                                bytearray(int_fmt.encode("utf8")))
            # global int_fmt
            int_fmt = ir.GlobalVariable(self.module, int_c_fmt.type, name="ifstr")
            int_fmt.linkage = 'internal'
            int_fmt.global_constant = True
            int_fmt.initializer = int_c_fmt

            # float
            float_fmt = "%.8g\n\0"
            float_c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(float_fmt)),
                                bytearray(float_fmt.encode("utf8")))
            # global float_fmt
            float_fmt = ir.GlobalVariable(self.module, float_c_fmt.type, name="ffstr")
            float_fmt.linkage = 'internal'
            float_fmt.global_constant = True
            float_fmt.initializer = float_c_fmt

            # read
            # float
            read_fmt = "%lf\0"
            read_c_fmt = ir.Constant(ir.ArrayType(ir.IntType(8), len(read_fmt)),
                                      bytearray(read_fmt.encode("utf8")))
            # global float_fmt
            read_fmt = ir.GlobalVariable(self.module, read_c_fmt.type, name="rfstr")
            read_fmt.linkage = 'internal'
            read_fmt.global_constant = True
            # print(read_float_fmt)
            read_fmt.initializer = read_c_fmt

            return int_fmt, float_fmt, read_fmt

        int_fmt, float_fmt, read_fmt = declare_fstrings()

        # create global var list
        if self.var_list is not None:
            self.var_list.generate(st, 'global', self.builder)

        # create functions
        if self.func_list is not None:
            self.func_list.generate(st, 'global', self.builder, int_fmt, float_fmt, read_fmt)

        # generate main function
        func_type = ir.FunctionType(ir.VoidType(), [], False)
        self.base_func = ir.Function(self.module, func_type, name="main")
        block = self.base_func.append_basic_block(name="entry")
        self.builder.position_at_end(block)
        int_fmt_arg, float_fmt_arg, read_fmt_arg = bitcast_fstrings(self.builder, int_fmt, float_fmt, read_fmt)
        self.block.generate(st, 'global', self.builder, self.base_func, int_fmt_arg, float_fmt_arg, read_fmt_arg)
        # print(block)

        self.builder.ret_void()


class Variable():
    def __init__(self, module, variable):
        # self.builder = builder
        self.module = module
        # self.printf = printf
        self.variable = variable
        # self.g_var = None
        self.st_var = None
        # self.i_var_ptr = None

    def to_string(self):
        print("variable", self.variable)

    def create_tree(self):
        parse_tree = Tree("Variable", [self.variable])
        return parse_tree

    def check(self, st, scope, assignment_st=None):
        var_name = self.variable
        if not (st.var_in_scope(var_name, scope) or st.var_in_scope(var_name, 'global')):
            # print('Var Scope Er')
            # print(var_name, scope)
            assert False, f'CompileError: Undefined variable {var_name} in block {scope}'
            # return False

        var = st.get_var_by_name(self.variable, scope)
        if var is None:
            var = st.get_var_by_name(self.variable, 'global')

        if assignment_st:
            # print(assignment_st)
            if not var.used_after_assign:
                if var.prev_assign_st:
                    # print(var.prev_assign_st)
                    var.prev_assign_st.not_used = True

            var.used_after_assign = False
            var.prev_assign_st = assignment_st
        else:
            var.used_after_assign = True
        return True

    def is_float(self, st, scope):
        var = st.get_var_by_name(self.variable, scope)
        if var is None:
            var = st.get_var_by_name(self.variable, 'global')

        if var.dtype == 'float':
            return True
        else:
            return False

    def is_const(self, st, scope):
        '''
        var = st.get_var_by_name(self.variable, scope)
        if var is None:
            var = st.get_var_by_name(self.variable, 'global')

        if var.is_const:
            return True
        else:
            return False
        '''
        return False


    '''
    def declare(self, st, scope):
        # print(g_var)
        self.g_var = ir.GlobalVariable(self.module, ir.IntType(32), self.variable + '_' + scope)
        # print(self.g_var)
        # return self.g_var

    def assign(self, val, st, scope):
        i = builder.store(val, self.g_var)

    def generate(self, st, scope, builder):
        # print(self.g_var)
        i = builder.load(self.g_var)
        return i
        
    '''

    def assign(self, st, scope, builder):
        if self.st_var is None:
            if st.var_in_scope(self.variable, scope):
                self.st_var = st.get_var_by_name(self.variable, scope)
            else:
                self.st_var = st.get_var_by_name(self.variable, 'global')

        return self.st_var.assign(builder, self.module)

    def generate(self, st, scope, builder):
        if self.st_var is None:
            if st.var_in_scope(self.variable, scope):
                self.st_var = st.get_var_by_name(self.variable, scope)
            else:
                self.st_var = st.get_var_by_name(self.variable, 'global')

        return self.st_var.generate(builder)

    '''
    def reload(self, st, scope):
        if self.st_var is None:
            self.st_var = st.get_var_by_name(self.variable, scope)

        self.st_var.reload(self.builder)
    '''


class FuncList():
    def __init__(self, func, func_list=None):
        # self.builder = builder
        # self.module = module
        # self.printf = printf
        # self.base_func = base_func
        self.func = func
        self.func_list = func_list
        # print("FuncList")

    def to_string(self):
        print("FuncList")
        self.func.to_string()

        if self.func_list is not None:
            self.func_list.to_string()

    def create_tree(self):
        if self.func_list is not None:
            parse_tree = Tree("FunctionsList", [self.func.create_tree(), self.func_list.create_tree()])
        else:
            # print(self.statement)
            parse_tree = Tree("FunctionsList", [self.func.create_tree()])
        return parse_tree

    def check(self, st, scope):
        if not self.func.check(st, scope):
            return False

        if self.func_list is not None:
            if not self.func_list.check(st, scope):
                return False

        return True

    def generate(self, st, scope, builder, int_fmt, float_fmt, read_fmt):
        self.func.generate(st, scope, builder, int_fmt, float_fmt, read_fmt)
        if self.func_list is not None:
            self.func_list.generate(st, scope, builder, int_fmt, float_fmt, read_fmt)


class Func():
    def __init__(self, module, name, dtype, block, args, var_list):
        # self.builder = builder
        self.module = module
        # self.printf = printf
        # self.base_func = base_func
        self.name = name
        self.dtype = dtype
        self.block = block
        self.args = args
        self.var_list = var_list
        # ST
        # sfunc = SFunc(self.name, self.dtype, args.get_list(), var_list.get_list())
        # ST.add_scope(sfunc)

    def to_string(self):
        print("Func", self.name, self.dtype)

        if self.args is not None:
            self.args.to_string()

        if self.var_list is not None:
            self.var_list.to_string()

        self.block.to_string()

    def create_tree(self):
        if self.var_list is not None and self.args is None:
            parse_tree = Tree("Func" + ' ' + self.name + ' ' + self.dtype, [self.var_list.create_tree(),
                                           self.block.create_tree()])

        elif self.var_list is None and self.args is not None:
            parse_tree = Tree("Func" + ' ' + self.name + ' ' + self.dtype, [self.args.create_tree(),
                                           self.block.create_tree()])

        elif self.var_list is not None and self.args is not None:
            parse_tree = Tree("Func" + ' ' + self.name + ' ' + self.dtype, [self.args.create_tree(),
                                           self.var_list.create_tree(),
                                           self.block.create_tree()])
        else:
            print(self.block)
            parse_tree = Tree("Function", [self.block.create_tree()])

        return parse_tree

    def check(self, st, scope):
        if not self.block.check(st, self.name):
            return False

        if self.args is not None:
            if not self.args.check(st, self.name):
                return False

        if self.var_list is not None:
            if not self.var_list.check(st, self.name):
                return False

        '''
        def ret_check():
            if self.block.contains_return():
                return True
            else:
                return False

        if not ret_check():
            assert False, "CompileError: Missing Return statament"
            # return False
        '''

        return True

    def generate(self, st, scope, builder, int_fmt, float_fmt, read_fmt):
        # print('ok')

        # create function declaration

        def create_llvm_args(args_list):

            result = []
            for arg in args_list:
                if arg[1] == 'int':
                    result.append(ir.IntType(32))
                elif arg[1] == 'float':
                    result.append(ir.DoubleType())

            return result

        if self.args is not None:
            tree_args = self.args.get_list()
            args_types = create_llvm_args(tree_args)
        else:
            tree_args = []
            args_types =[]

        if self.dtype == 'int':
            return_type = ir.IntType(32)
        elif self.dtype == 'float':
            return_type = ir.DoubleType()
        else:
            return_type = None

        function_type = ir.FunctionType(return_type, args_types)

        ir_func = ir.Function(self.module, function_type, name=self.name)

        st_func = st.get_func_by_name(self.name)
        st_func.set_ir_func(ir_func)

        entry_block = ir_func.append_basic_block('entry')
        builder.position_at_end(entry_block)

        # generate arg list
        def link_fact_args(args_list, ir_args):
            for i in range(len(args_list)):
                st_var = st.get_var_by_name(args_list[i][0], self.name)
                builder.store(ir_args[i], st_var.assign(builder, self.module))

        if self.args is not None:
            link_fact_args(tree_args, ir_func.args)

        # generate var list
        if self.var_list is not None:
            self.var_list.generate(st, self.name, builder)

        # generate block
        int_fmt_arg, float_fmt_arg, read_fmt_arg = bitcast_fstrings(builder, int_fmt, float_fmt, read_fmt)
        self.block.generate(st, self.name, builder, ir_func, int_fmt_arg, float_fmt_arg, read_fmt_arg)

        if not builder.block.is_terminated:
            builder.ret(ir.Constant(return_type, 0))


class ParamList():
    def __init__(self, param, param_list=None):
        # self.builder = builder
        # self.module = module
        # self.printf = printf
        self.param = param
        self.param_list = param_list

    def to_string(self):
        print("ParamList")
        self.param.to_string()

        if self.param_list is not None:
            self.param_list.to_string()

    def create_tree(self):
        if self.param_list is not None:
            parse_tree = Tree("ParamList", [self.param.create_tree(), self.param_list.create_tree()])
        else:
            parse_tree = Tree("ParamList", [self.param.create_tree()])

        return parse_tree

    def get_list(self):
        if self.param_list is None:
            return [self.param]
        else:
            return [self.param] + self.param_list.get_list()

    def check(self, st, scope, called):
        called_function = st.get_func_by_name(called)
        fact_args = self.get_list()
        declared_args = called_function.args
        if len(declared_args) != len(fact_args):
            assert False, f'CompileError: Incorrect number of parameters when call {called_function}'
            # return False
        for i in range(len(declared_args)):
            if declared_args[i][1] == 'int' and fact_args[i].is_float(st, scope):
                # assert False, f'CompileError: Incorrect types of parameters when call {called_function}'
                # return False
                return True
            if not fact_args[i].check(st, scope):
                return False

        return True

    def generate(self, st, scope, builder, called):
        called_function = st.get_func_by_name(called)
        fact_args = self.get_list()
        declared_args = called_function.args

        ir_params = []
        for i in range(len(fact_args)):
            param = fact_args[i]
            ir_param = param.generate(st, scope, builder)
            if declared_args[i][1] == 'int' and param.is_float(st, scope):
                ir_param = builder.fptosi(ir_param, ir.IntType(32))
            if declared_args[i][1] == 'float' and not param.is_float(st, scope):
                ir_param = builder.sitofp(ir_param, ir.DoubleType)
            ir_params.append(ir_param)

        return ir_params


class FuncCall():
    def __init__(self, name, param_list=None):
        # self.builder = builder
        # self.module = module
        # self.printf = printf
        self.name = name
        self.param_list = param_list

    def to_string(self):
        print("FuncCall", self.name)

        if self.param_list is not None:
            self.param_list.to_string()

    def create_tree(self):
        if self.param_list is not None:
            parse_tree = Tree("FuncCall" + ' ' + self.name, [self.param_list.create_tree()])
        else:
            parse_tree = "FuncCall" + ' ' + self.name

        return parse_tree

    def check(self, st, scope):
        if st.get_func_by_name(self.name) is None:
            return False

        if self.param_list is not None:
            if not self.param_list.check(st, scope, self.name):
                return False

        return True

    def is_float(self, st, scope):
        if st.get_func_by_name(self.name).dtype == 'int':
            return False
        else:
            return True

    def generate(self, st, scope, builder):
        st_func = st.get_func_by_name(self.name)
        ir_func = st_func.get_ir_func()
        if self.param_list is not None:
            ir_params = self.param_list.generate(st, scope, builder, self.name)
        else:
            ir_params = []

        ir_call = builder.call(ir_func, ir_params)
        return ir_call

    def is_const(self, st, scope):
        return False


class Return(Statement):
    def to_string(self):
        print("ReturnStatement")
        self.block1.to_string()

    def create_tree(self):
        parse_tree = Tree("ReturnStatement", [self.block1.create_tree()])
        return parse_tree

    def check(self, st, scope):
        if scope == 'global':
            assert False, f'CompileError: Return statement out of function'
            # return False

        if not self.block1.check(st, scope):
            return False

        called_function = st.get_func_by_name(scope)
        if called_function.dtype == 'int' and self.block1.is_float(st, scope):
            # assert False, f"CompileError: Can't return float value from int function {called_function.name}"
            # return False
            return True

        return True

    def generate(self, st, scope, builder):
        # print('ok2')
        called_function = st.get_func_by_name(scope)
        ret_value = self.block1.generate(st, scope, builder)
        if called_function.dtype == 'int' and self.block1.is_float(st, scope):
            ret_value = builder.fptosi(ret_value, ir.IntType(32))
        if called_function.dtype == 'float' and not self.block1.is_float(st, scope):
            ret_value = builder.fptosi(ret_value, ir.DoubleType())
        builder.ret(ret_value)


class Break():
    def __init__(self):
        # self.module = module
        pass

    def create_tree(self):
        return 'Break'

    def check(self, st, scope, in_cycle):
        assert in_cycle, f'CompileError: Break statement out of cycle'
        return True

    def generate(self, st, scope, builder, end_block):
        builder.branch(end_block)


class Continue():
    def __init__(self):
        # self.module = module
        pass

    def create_tree(self):
        return 'Continue'

    def check(self, st, scope, in_cycle=False):
        assert in_cycle, f'CompileError: Continue statement out of cycle'
        return True

    def generate(self, st, scope, builder, head_block):
        # print(head_block)
        builder.branch(head_block)


class ReadStatement():
    def __init__(self, variable, scanf):
        # self.module = module
        self.variable = variable
        self.scanf = scanf

    def create_tree(self):
        return Tree('ReadStatement', [self.variable.create_tree()])

    def check(self, st, scope):
        if not self.variable.check(st, scope):
            return False

        return True

    def generate(self, st, scope, builder, read_fmt_arg):
        ir_var = self.variable.assign(st, scope, builder)
        # Call Scanf Function
        # print(read_int_fmt_arg)
        if not self.variable.is_float(st, scope):
            tmp = builder.alloca(ir.DoubleType())
            builder.call(self.scanf, [read_fmt_arg, tmp])
            builder.store(builder.fptosi(builder.load(tmp), ir.IntType(32)), ir_var)
        else:
            builder.call(self.scanf, [read_fmt_arg, ir_var])

