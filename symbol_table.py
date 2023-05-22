from llvmlite import ir


class SVar:
    def __init__(self, name, dtype, scope):
        self.name = name
        self.dtype = dtype
        self.scope = scope
        self.ir_var = None
        self.loaded = None
        self.prev_assign_st = None
        self.used_after_assign = True
        # self.used = False

    def assign(self, builder, module):
        if self.ir_var is None:
            if self.scope != 'global':
                if self.dtype == 'int':
                    self.ir_var = builder.alloca(ir.IntType(32))
                else:
                    self.ir_var = builder.alloca(ir.DoubleType())
            else:
                # print(self.name)
                if self.dtype == 'int':
                    self.ir_var = ir.GlobalVariable(module, ir.IntType(32), self.name)
                else:
                    self.ir_var = ir.GlobalVariable(module, ir.DoubleType(), self.name)

        return self.ir_var

    # def reload(self, builder):
        # self.loaded = builder.load(self.i_var)

    def generate(self, builder):
        i = builder.load(self.ir_var)
        return i
        # return self.loaded

    def init0(self, builder, module):
        if self.scope != 'global':
            if self.dtype == 'int':
                builder.store(ir.Constant(ir.IntType(32), 0), self.assign(builder, module))
            else:
                builder.store(ir.Constant(ir.DoubleType(), 0.0), self.assign(builder, module))
        else:
            # print(self.ir_var)
            self.assign(builder, module)
            # print(self.ir_var is None)
            if self.dtype == 'int':
                # print(self.ir_var.initializer)
                self.ir_var.initializer = ir.Constant(ir.IntType(32), 0)
            else:
                self.ir_var.initializer = ir.Constant(ir.DoubleType(), 0.0)
        # self.reload(builder)

    # def create_global(self, builder):


class SFunc:
    def __init__(self, name, dtype, args, variables):
        self.name = name
        self.dtype = dtype
        self.args = args
        self.variables = variables
        self.ir_func = None

    def set_ir_func(self, ir_func):
        self.ir_func = ir_func

    def get_ir_func(self):
        return self.ir_func


class SymbolTable:
    def __init__(self):
        self.table = {'global': [[], []]}
        self.funcs = {}
        # self.variables = {}

    def add_scope(self, new_scope):
        # if new_scope.name in self.funcs:
            #raise ValueError
        assert new_scope.name not in self.funcs, f"CompileError: Repeated {new_scope.name} (function) definition"

        self.table.update({new_scope.name: [[], []]})
        self.funcs.update({new_scope.name: new_scope})

    def add_var(self, new_var):
        # if new_var.name in self.table[new_var.scope][0]:
            # raise ValueError
        assert new_var.name not in self.table[new_var.scope][0], f"CompileError: Repeated {new_var.name} (variable) definition"

        self.table[new_var.scope][0].append(new_var.name)
        self.table[new_var.scope][1].append(new_var)
        # self.variables.update({new_var.name, new_var})

    def __getitem__(self, item):
        return self.table[item]

    def var_in_scope(self, var_name, scope_name):
        if var_name in self[scope_name][0]:
            return True
        else:
            return False

    def get_func_by_name(self, func_name):
        if func_name in self.funcs:
            return self.funcs[func_name]
        else:
            # raise ValueError("Incorrect name", func_name)
            return None

    def get_var_by_name(self, var_name, scope_name):
        if self.var_in_scope(var_name, scope_name):
            return self.table[scope_name][1][self.table[scope_name][0].index(var_name)]
        else:
            # raise ValueError("Incorrect name", func_or_var_name)
            return None


    def fill(self, program):
        def fill_from_var_list(var_list, scope):
            variables = var_list.get_list()
            for variable in variables:
                new_var = SVar(variable[0], variable[1], scope)
                self.add_var(new_var)

        if program.var_list is not None:
            fill_from_var_list(program.var_list, 'global')

        def fill_from_func_list(func_list):
            function = func_list.func
            args = None if function.args is None else function.args.get_list()
            variables = None if function.var_list is None else function.var_list.get_list()

            new_function = SFunc(function.name, function.dtype, args, variables)
            self.add_scope(new_function)

            if function.args is not None:
                fill_from_var_list(function.args, function.name)
            if function.var_list is not None:
                fill_from_var_list(function.var_list, function.name)

            if func_list.func_list is not None:
                fill_from_func_list(func_list.func_list)

        if program.func_list is not None:
            fill_from_func_list(program.func_list)
