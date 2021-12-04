#!/usr/bin/env nix-shell
#!nix-shell -i python -p llvmPackages.clang-unwrapped.python

import sys
import os
 
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/bindings/python')
from clang.cindex import Config, Index, TranslationUnit

header = sys.argv[1]
namespace = sys.argv[2]
ns = ''
cl = ''

translation_unit = Index.create().parse(header, options = TranslationUnit.PARSE_SKIP_FUNCTION_BODIES)

def dump(cursor, indent=0):
    global ns,cl
    if cursor.kind.name == 'NAMESPACE' :
            ns = cursor.spelling;
    if cursor.kind.name == 'CLASS_DECL' :
            cl = cursor.spelling;
    
    if cursor.kind.name == 'CXX_METHOD' and ns == namespace:
            print('- name: ' + cursor.spelling)
            print('  type: ' + cursor.type.spelling)
            print('  namespace: ' + ns + '::' + cl)
    if cursor.kind.name == 'FUNCTION_DECL' and ns == namespace:
            print('- name: ' + cursor.spelling)
            print('  type: ' + cursor.type.spelling)
            print('  namespace: ' + ns)
    
    for child in cursor.get_children():
    	dump(child, indent+1)
 
dump(translation_unit.cursor)
