#!/usr/bin/env nix-shell
#!nix-shell -i python -p llvmPackages.clang-unwrapped.python

import sys
import os
 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/bindings/python')
from clang.cindex import Config, Index, TranslationUnit

header = sys.argv[1]
namespace = sys.argv[2]
classname = sys.argv[3]
ns = ''
cl = ''

translation_unit = Index.create().parse(header, options = TranslationUnit.PARSE_SKIP_FUNCTION_BODIES)

def dump(cursor, indent=0):
    global ns,cl
    if cursor.kind.name == 'NAMESPACE' :
            ns = cursor.spelling;
    if cursor.kind.name == 'CLASS_DECL' and ns == namespace:
            cl = cursor.spelling;
            if classname == cl: 
                print('signature: ' + cl)
                print('cppname: ' + ns + '::' + cl)
                print('hsname: ' + cl)
                print('headers:')
                print('- ' + header)
                print('functions: []')
                print('constructors:')
                print('methods:')
    
    if cursor.kind.name == 'CXX_METHOD' and ns == namespace and cl == classname:
            print('- name: ' + cursor.spelling)
            ts = cursor.type.spelling
            pos = ts.find('(')
            print('- ' + cursor.spelling + ' ' + ts[pos:] + ' -> ' + ts[0:pos-1])
    if cursor.kind.name == 'FUNCTION_DECL' and ns == namespace and cl == classname:
            print('- name: ' + cursor.spelling)
            ts = cursor.type.spelling
            pos = ts.find('(')
            print('- ' + cursor.spelling + ' ' + ts[pos:] + ' -> ' + ts[0:pos-1])
    
    for child in cursor.get_children():
    	dump(child, indent+1)
 
dump(translation_unit.cursor)
