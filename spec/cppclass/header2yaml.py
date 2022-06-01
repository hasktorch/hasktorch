#!/usr/bin/env python

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/bindings/python')
from clang.cindex import Config, Index, TranslationUnit

ns = ''
cl = ''

def dump(cursor, indent, args):
    global ns,cl
    if cursor.kind.name == 'NAMESPACE' :
            ns = cursor.spelling;
    if cursor.kind.name == 'CLASS_DECL' and ns == args.namespace:
            cl = cursor.spelling;
            if args.classname == cl: 
                print('signature: ' + cl)
                print('cppname: ' + ns + '::' + cl)
                print('hsname: ' + cl)
                print('headers:')
                print('- ' + args.header)
                print('functions: []')
                print('constructors:')
                print('methods:')
    
    if cursor.kind.name == 'CXX_METHOD' and ns == args.namespace and cl == args.classname:
            #print('- name: ' + cursor.spelling)
            ts = cursor.type.spelling
            pos = ts.find('(')
            print('- ' + cursor.spelling + ' ' + ts[pos:] + ' -> ' + ts[0:pos-1])
    if cursor.kind.name == 'FUNCTION_DECL' and ns == args.namespace and cl == args.classname:
            #print('- name: ' + cursor.spelling)
            ts = cursor.type.spelling
            pos = ts.find('(')
            print('- ' + cursor.spelling + ' ' + ts[pos:] + ' -> ' + ts[0:pos-1])
    if args.verbose:
        print((" " * indent)  + "ns:" + ns + ", cl:" + cl + ", kind:" + cursor.kind.name + ", cursor.spelling:" + cursor.spelling)
            
    for child in cursor.get_children():
    	dump(child, indent+1, args)
 

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='Generate type specs of c++-class to generate interface for haskell', add_help=add_help)

    parser.add_argument('--header', default=None, help='A path of c++ header file')
    parser.add_argument('--namespace', default=None, help='A namespace of class to generate interface for haskell')
    parser.add_argument('--classname', default=None, help='A classname to generate interface for haskell')
    parser.add_argument('--verbose', default=False, help='', action="store_true")
    return parser

def main(args):
    translation_unit = Index.create().parse(args.header, args = ["-DUSE_C10D_NCCL", "-DUSE_C10D_MPI"], options = TranslationUnit.PARSE_SKIP_FUNCTION_BODIES)
    dump(translation_unit.cursor, 0, args)

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
