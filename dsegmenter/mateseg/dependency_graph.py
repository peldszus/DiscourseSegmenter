#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

'''
Created on 03.01.2015

@author: Andreas Peldszus
'''

import codecs
from nltk.parse.dependencygraph import DependencyGraph as NLTKDependencyGraph


##############################################################################
# DEPENDENCY GRAPH HELPERS
##############################################################################

HEAD = "head"
DEPS = "deps"
WORD = "word"
REL = "rel"
TAG = "tag"
ADDRESS = "address"

TOP_TAG_LABEL = 'TOP'
TOP_RELATION_LABEL = 'ROOT'
'''The label of relations to the root node in the mate dep. parser output.'''


class DependencyGraph(NLTKDependencyGraph):

    def words(self):
        '''yields all words except the implicit root node in linear order'''
        for address, node in sorted(self.nodes.items()):
            if node['tag'] != 'TOP':
                yield node['word']

    def subgraphs(self, exclude_root=False):
        '''yields all nodes in linear order'''
        for address, node in sorted(self.nodes.items()):
            if exclude_root and node['tag'] == 'TOP':
                continue
            else:
                yield node

    def get_dependencies_simple(self, address):
        '''returns a sorted list of the addresses of all dependencies of the
           node at the specified address'''
        deps_dict = self.nodes[address].get('deps', {})
        return sorted([e for l in deps_dict.values() for e in l])

    def address_span(self, start_address):
        '''returns the addresses of nodes (im)mediately depending on the given
           starting address in a dependency graph, except for the root node'''
        worklist = [start_address]
        addresses = []
        while len(worklist) != 0:
            address = worklist.pop(0)
            addresses.append(address)
            for _rel, deps in self.nodes[address]['deps'].items():
                worklist.extend(deps)
        return sorted(addresses)

    def token_span(self, start_address=0):
        '''returns the words (im)mediately depending on the given address in a
           dependency graph in correct linear order, except for the root node
        '''
        addresses = self.address_span(start_address)
        return [self.nodes[address]['word']
                for address in sorted(addresses) if address != 0]

    def is_valid_parse_tree(self):
        root = self.get_dependencies_simple(0)
        if len(root) < 1:
            print "Warning: No root address"
            return False
        if len(root) > 1:
            print "Warning: More than one root address"
            return False
        # TODO: Add more constraints
        return True


def transform_conll_data(data):
    '''transforms conll data outputted by the mate parser to valid
       conll 2007 format'''
    out = []
    for line in data.splitlines():
        out.append(transform_line(line))
    return '\n'.join(out)


def transform_line(line):
    '''transforms a conll line outputted by the mate parser to a valid
       conll 2007 format line'''
    if line.strip() == '':
        return ''
    else:
        f = line.split('\t')
        # escape parenthesis
        token = f[1]
        if token == '(':
            token = '-OP-'
        elif token == ')':
            token = '-CP-'
        # The nltk v3 implementation of dependency graphs needs an explicit
        # root relation label. Mate's output uses '--' as a label for relations
        # to the root, but also for punctuations. We thus translate the
        # relation label to 'ROOT'.
        if f[9] == '0':
            f[11] = 'ROOT'
        return '\t'.join([f[0], token, f[3], f[5], f[5], f[7], f[9], f[11],
                          '_', '_'])


def number_tokens_of_dependency_graphs(list_of_dependency_graphs):
    '''prefixes all tokens in a list of dependency graphs with a running number
       starting from 0'''
    deptree_leaf_counter = 0
    for depgraph in list_of_dependency_graphs:
        for node in depgraph.subgraphs(exclude_root=True):
            node['word'] = (deptree_leaf_counter, node['word'])
            deptree_leaf_counter += 1
    return list_of_dependency_graphs


def read_deptree_file(fn):
    '''reads mate parser output and returns a list of dependency graphs of the
       parsed sentences'''
    with codecs.open(fn, 'r', 'utf-8') as f:
        s = transform_conll_data(f.read())
        l = [DependencyGraph(sentence, top_relation_label=TOP_RELATION_LABEL)
             for sentence in s.split('\n\n') if sentence]
        return number_tokens_of_dependency_graphs(l)
