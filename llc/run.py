import os

# filename = input('Input filename: \n')
filename = "my_output"

os.system('llc -filetype=obj ' + filename + '.ll')
print('LLC - OK')

os.system('gcc ' + filename + '.obj' + ' -no-pie -o out.exe' )
print('GCC - OK')

print('RUN:')
os.system('out.exe')
