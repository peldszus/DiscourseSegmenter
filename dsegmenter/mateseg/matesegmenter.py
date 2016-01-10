#!/usr/bin/env python2.7
# -*- mode: python; coding: utf-8; -*-

'''
Created on 03.01.2015

@author: Andreas Peldszus
'''

import sys
import os
import argparse
from itertools import chain
from collections import defaultdict
from functools import partial

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.cross_validation import KFold
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals import joblib

from .dependency_graph import read_deptree_file
from .segmentation_tree import read_segtree_file, generate_subtrees_from_forest
from ..treeseg import (TreeSegmenter, DiscourseSegment, DEPENDENCY,
                       DEFAULT_SEGMENT)
from ..treeseg.treesegmenter import HEAD, WORD, REL, TAG, NO_MATCH_STRING
from ..treeseg.constants import GREEDY
from ..bparseg.align import nw_align


number_of_folds = 10
punct_tags = ['$.', '$,']
feature_not_found = '[NONE]'


def gen_features_for_segment(dep_graph, trg_adr):
    ''' ugly feature extraction code  ;) '''

    nodes = list(dep_graph.subgraphs(exclude_root=False))
    nl = {node['address']: node for node in nodes}
    assert len(nodes) == len(nl)

    if trg_adr >= len(nl):
        return {}
    seg_adr_span = dep_graph.address_span(trg_adr)

    # get relation, this word and pos
    rel = nl[trg_adr][REL] if REL in nl[trg_adr] else feature_not_found
    this_word = nl[trg_adr][WORD][1] if WORD in nl[trg_adr] and nl[trg_adr][WORD] != None else feature_not_found
    this_pos = nl[trg_adr][TAG] if TAG in nl[trg_adr] else feature_not_found

    # get head word and pos
    head_adr = nl[trg_adr][HEAD] if HEAD in nl[trg_adr] else None
    if head_adr is not None:
        head_word = nl[head_adr][WORD][1] if WORD in nl[head_adr] and nl[head_adr][WORD] != None else feature_not_found
        head_pos = nl[head_adr][TAG] if TAG in nl[head_adr] else feature_not_found
    else:
        head_word = head_pos = feature_not_found

    # get first and last word from segment
    first_adr = seg_adr_span[0]
    first_word = nl[first_adr][WORD][1] if nl[first_adr][WORD] != None else feature_not_found
    last_adr = seg_adr_span[-1]
    last_word = nl[last_adr][WORD][1] if nl[last_adr][WORD] != None else feature_not_found

    # get words left and right from segment
    left_adr = seg_adr_span[0] - 1
    left_word = nl[left_adr][WORD][1] if len(nl) > left_adr > 0 and WORD in nl[left_adr] and nl[left_adr][WORD] != None else feature_not_found
    right_adr = seg_adr_span[-1] + 1
    right_word = nl[right_adr][WORD][1] if len(nl) > right_adr > 0 and WORD in nl[right_adr] and nl[right_adr][WORD] != None else feature_not_found

    # get segment length
    length_abs = len(seg_adr_span)

    # get number of punctuation in segment
    punct_count = sum([1 if adr in nl and TAG in nl[adr] and nl[adr][TAG] in punct_tags else 0 for adr in seg_adr_span])

    # resulting feature dictionary
    r = {
        'rel': rel,
        'head_word': head_word,
        'head_pos': head_pos,
        'this_word': this_word,
        'this_pos': this_pos,
        'rel+head_pos+this_pos': '+'.join([rel, head_pos, this_pos]),
        'rel+head_pos': '+'.join([rel, head_pos]),
        'rel+this_pos': '+'.join([rel, this_pos]),
        'head_pos+this_pos': '+'.join([head_pos, this_pos]),
        'first_word': first_word,
        'last_word': last_word,
        'left_word': left_word,
        'right_word': right_word,
        'length_abs': length_abs,
        'punct_count': punct_count,
    }

    # simply add all words for this segment
    for adr in seg_adr_span:
        if nl[adr][WORD] is not None:
            word = nl[adr][WORD][1]
            r['word_%s' % word] = 1

    return r


def word_access(x):
    if x is None:
        return ''
    else:
        return x[1]


def substitution_costs(c1, c2):
    '''defines the costs of substitutions for the alignment'''
    if c1[-1] == c2[-1]:
        return 2
    else:
        return -3


def get_testing_observations(dep_trees, text_id):
    for sentence_index, dep_tree in enumerate(dep_trees):
        for node in dep_tree.subgraphs(exclude_root=True):
            yield (text_id, sentence_index, node['address'], dep_tree)


def get_observations(seg_trees, dep_trees, text_id):
    # pregenerate all dependency subgraphs
    items = []
    for sentence_index, dep_tree in enumerate(dep_trees):
        for node in dep_tree.subgraphs(exclude_root=True):
            tokset = set(dep_tree.token_span(node['address']))
            items.append((text_id, sentence_index, node['address'], dep_tree,
                          tokset))

    # match tokenization first
    seg_tokens = list(chain.from_iterable([
        tree.leaves() for tree in seg_trees]))
    dep_tokens = list(chain.from_iterable([
        dg.words() for dg in dep_trees]))
    unequal_tokenizations = False
    if seg_tokens != dep_tokens:
        unequal_tokenizations = True
        aligned = nw_align(dep_tokens, seg_tokens,
                           substitute=substitution_costs, keep_deleted=True)
        # make a translation
        seg_to_dep_tok = {}
        for dep_index, seg_index_list in enumerate(aligned):
            for seg_index in seg_index_list:
                seg_to_dep_tok[seg_tokens[seg_index]] = dep_tokens[dep_index]

    # match every dep_tree subgraphs with all seg_tree non-terminals
    for text_id, sentence_index, address, dep_tree, tokset in items:
        found_match = False
        for seg_sub_tree in generate_subtrees_from_forest(seg_trees):
            node = seg_sub_tree.label()
            if node is None or node == "":
                print "Warning: Empty node.", text_id, sentence_index
            if unequal_tokenizations:
                seg_leaves = set([seg_to_dep_tok[leaf]
                                 for leaf in seg_sub_tree.leaves()])
            else:
                seg_leaves = set(seg_sub_tree.leaves())
            if seg_leaves == tokset:
                found_match = True
                yield (text_id, sentence_index, address, dep_tree, node)
                break
        if not found_match:
            yield (text_id, sentence_index, address, dep_tree, NO_MATCH_STRING)


def _cnt_stat(a_gold_segs, a_pred_segs):
    """Estimate the number of true positives, false positives, and false negatives

    @param a_gold_segs - gold segments
    @param a_pred_segs - predicted segments

    @return 3-tuple with true positives, false positives, and false negatives

    """
    tp = fp = fn = 0
    for gs, ps in zip(a_gold_segs, a_pred_segs):
        gs = gs.lower()
        ps = ps.lower()
        if gs == "none":
            if ps != "none":
                fp += 1
        elif gs == ps:
            tp += 1
        else:
            fn += 1
    return tp, fp, fn


def decision_function(node, tree, predictions=None, items=None, text=None,
                      sentence=None):
    '''decision function for the tree segmenter'''
    index = items.index((text, sentence, node['address']))
    pred = predictions[index]
    if pred == NO_MATCH_STRING and 'head' in node and node['head'] == 0:
        # The classifier did not recognize sentence top as a segment, so we
        # enforce a labelling with the default segment type.
        return DEFAULT_SEGMENT
    else:
        return pred


def predict_segments_from_vector(parses, predictions, items, text):
    '''construct a tree of discourse segments from the predictions
       based on dependency graphs nodes'''
    segmenter = TreeSegmenter(a_type=DEPENDENCY)
    all_segments = []
    for sentence, dep_graph in enumerate(parses):
        if dep_graph.is_valid_parse_tree():
            dec_func = partial(decision_function,
                               predictions=predictions, items=items,
                               text=text, sentence=sentence)
            segments = segmenter.segment(
                dep_graph, a_predict=dec_func,
                a_word_access=word_access, a_strategy=GREEDY,
                a_root_idx=dep_graph.root['address'])
        else:
            # make a simple sentence segment for invalid parse trees
            leaves = [(i, word) for i, (_, word) in
                      enumerate(dep_graph.words(), 1)]
            dseg = DiscourseSegment(a_name=DEFAULT_SEGMENT, a_leaves=leaves)
            segments = [(0, dseg)]

        # set segment index
        segment = segments[0][1]
        all_segments.append((sentence, segment))

    return DiscourseSegment(a_name='TEXT', a_leaves=all_segments)


def classify(texts, seg, dep, out_folder):
    # extract all features
    print "Extracting features from all texts..."
    texts = np.array(sorted(list(texts)))
    all_observations = list(chain.from_iterable([
        get_observations(seg[text], dep[text], text) for text in texts
    ]))
    all_X = {}
    all_y = {}
    items_per_text = defaultdict(list)
    for text_id, sentence_index, address, dep_tree, class_ in all_observations:
        all_X[(text_id, sentence_index, address)] = gen_features_for_segment(
            dep_tree, address)
        all_y[(text_id, sentence_index, address)] = class_
        items_per_text[text_id].append((text_id, sentence_index, address))

    # cross validating
    folds = KFold(len(texts), number_of_folds)  # , shuffle=True)
    macro_F1s = []
    micro_F1s = []
    tp = fp = fn = tp_i = fp_i = fn_i = 0
    for i, (train, test) in enumerate(folds):
        # creating fold
        print "# FOLD", i
        train_texts = texts[train]
        train_items = list(chain.from_iterable([items_per_text[text]
                                                for text in train_texts]))
        train_X = [all_X[item] for item in train_items]
        train_y = [all_y[item] for item in train_items]

        test_texts = texts[test]
        test_items = list(chain.from_iterable([items_per_text[text]
                                               for text in test_texts]))
        test_X = [all_X[item] for item in test_items]
        test_y = [all_y[item] for item in test_items]

        # specify pipeline
        print "  training on %d items..." % len(train_y)
        pipeline = Pipeline([
            ('vectorizer', DictVectorizer()),
            ('var_filter', VarianceThreshold()),
            ('LinearSVC', LinearSVC(class_weight='auto'))])
        pipeline.fit(train_X, train_y)
        print "  extracted %d features using the dict vectorizer." % \
            len(pipeline.named_steps['vectorizer'].get_feature_names())

        # (test on testset to give an internal number)
        print "  testing on %d items..." % len(test_y)
        pred_y = pipeline.predict(test_X)

        tp_i, fp_i, fn_i = _cnt_stat(test_y, pred_y)
        tp += tp_i
        fp += fp_i
        fn += fn_i
        _p, _r, macro_f1, _s = precision_recall_fscore_support(
            test_y, pred_y, average='macro', pos_label=None)
        _p, _r, micro_f1, _s = precision_recall_fscore_support(
            test_y, pred_y, average='micro', pos_label=None)
        macro_F1s.append(macro_f1)
        micro_F1s.append(micro_f1)
        print "  Macro F1 = %3.1f, Micro F1 = %3.1f" % \
            (100 * macro_f1, 100 * micro_f1)

        # send predicted test set to tree segmenter
        print "  writing predictions as bracket tree..."
        for text in test_texts:
            discourse_tree = predict_segments_from_vector(
                dep[text], pred_y, test_items, text)
            print text
            with open(out_folder + '/' + text + '.tree', 'w') as fout:
                fout.write(str(discourse_tree))

    print "# Average Macro F1 = %3.1f +- %3.2f" % (100 * np.mean(macro_F1s),
                                                   100 * np.std(macro_F1s))
    print "# Average Micro F1 = %3.1f +- %3.2f" % (100 * np.mean(micro_F1s),
                                                   100 * np.std(micro_F1s))
    if tp or fp or fn:
        print "# F1_{tp,fp} %.2f" % (2. * tp / (2. * tp + fp + fn) * 100)
    else:
        print "# F1_{tp,fp} 0. %"


def train_and_save_model(texts, seg, dep, out_folder):
    # extract all features
    print "Extracting features from all texts..."
    texts = np.array(sorted(list(texts)))
    all_observations = list(chain.from_iterable([
        get_observations(seg[text], dep[text], text) for text in texts]))
    all_X = {}
    all_y = {}
    items_per_text = defaultdict(list)
    for text_id, sentence_index, address, dep_tree, class_ in all_observations:
        all_X[(text_id, sentence_index, address)] = gen_features_for_segment(
            dep_tree, address)
        all_y[(text_id, sentence_index, address)] = class_
        items_per_text[text_id].append((text_id, sentence_index, address))

    train_texts = texts
    train_items = list(chain.from_iterable([items_per_text[text]
                       for text in train_texts]))
    train_X = [all_X[item] for item in train_items]
    train_y = [all_y[item] for item in train_items]

    # specify pipeline
    print "Training on %d items..." % len(train_y)
    pipeline = Pipeline([
        ('vectorizer', DictVectorizer()),
        ('var_filter', VarianceThreshold()),
        ('LinearSVC', LinearSVC(class_weight='auto'))])
    pipeline.fit(train_X, train_y)

    print "Saving model..."
    joblib.dump(pipeline, out_folder + '/mate-segmenter.pkl',
                compress=1, cache_size=1e9)

    print "Done."


def test(dep, model_file, out_folder):
    print "Loading model..."
    pipeline = joblib.load(model_file)

    # extract all features
    print "Extracting features from all texts..."
    texts = np.array(sorted(dep.keys()))

    all_observations = list(chain.from_iterable([
        get_testing_observations(dep[text], text) for text in texts]))
    all_X = {}
    items_per_text = defaultdict(list)
    for text_id, sentence_index, address, dep_tree in all_observations:
        all_X[(text_id, sentence_index, address)] = gen_features_for_segment(
            dep_tree, address)
        items_per_text[text_id].append((text_id, sentence_index, address))

    # predict
    print "Predicting..."
    test_items = list(chain.from_iterable([items_per_text[text]
                      for text in texts]))
    test_X = [all_X[item] for item in test_items]
    pred_y = pipeline.predict(test_X)

    # send predicted test set to tree segmenter
    print "  writing predictions as bracket tree..."
    for text in texts:
        discourse_tree = predict_segments_from_vector(
            dep[text], pred_y, test_items, text)
        print text
        with open(out_folder + '/' + text + '.tree', 'w') as fout:
            fout.write(str(discourse_tree))

    print "Done."


def main():
    # initialize argument parser
    aparser = argparse.ArgumentParser(
        description=("A discourse segmentation model to be trained and ",
                     "tested on dependency parses."))
    aparser.add_argument(
        "mode", help="mode", choices=['eval', 'train', 'test'])
    aparser.add_argument(
        "in_seg", help=("input folder for segmentation files ",
                        "(will be ignored in test mode)"))
    aparser.add_argument(
        "in_dep", help="input folder for mate dependencies")
    aparser.add_argument(
        "out_folder", help="output folder for predictions or models")
    aparser.add_argument(
        "--model", help="model to use for prediction", nargs=1)
    args = aparser.parse_args()

    seg_folder = args.in_seg
    dep_folder = args.in_dep
    out_folder = args.out_folder
    file_suffix_seg = '.tree'
    file_suffix_dep = '.parsed.conll'

    mode = args.mode

    if mode in ['eval', 'train']:
        # find files
        print 'Finding segmentation files...',
        seg_files = sorted([f for f in os.listdir(seg_folder)
                            if f.endswith(file_suffix_seg)])
        print 'found %d.' % len(seg_files)
        print 'Finding mate parse files...',
        dep_files = sorted([f for f in os.listdir(dep_folder)
                            if f.endswith(file_suffix_dep)])
        print 'found %d.' % len(dep_files)

        # check text alignment
        print 'Checking text alignment...',
        seg_texts = set([fn.split('.')[0] for fn in seg_files])
        dep_texts = set([fn.split('.')[0] for fn in dep_files])
        if seg_texts == dep_texts:
            print 'passed.'
        else:
            print 'failed.'
            print 'Texts in segmentation folder:', ','.join([t for t in sorted(seg_texts)])
            print 'Texts in dependency folder:', ','.join([t for t in sorted(dep_texts)])
            sys.exit(-1)

        # initialize base data structures
        texts = seg_texts
        seg = {t: [] for t in texts}
        dep = {t: [] for t in texts}

        # load all files
        print 'Loading input files...',
        for text in texts:
            # load segmentation
            seg[text] = read_segtree_file(seg_folder + '/' + text + file_suffix_seg)
            dep[text] = read_deptree_file(dep_folder + '/' + text + file_suffix_dep)
        print 'done.'

        if mode == 'eval':
            classify(texts, seg, dep, out_folder)
        elif mode == 'train':
            train_and_save_model(texts, seg, dep, out_folder)

    elif mode == 'test':
        if len(args.model) != 1 or args.model[0] == None or args.model[0] == '':
            print "Specify a model to test."

        # find files
        print 'Finding mate parse files...',
        dep_files = sorted([f for f in os.listdir(dep_folder)
                            if f.endswith(file_suffix_dep)])
        dep_texts = set([fn.rsplit(file_suffix_dep, 1)[0] for fn in dep_files])
        dep = {t: [] for t in dep_texts}
        print 'found %d.' % len(dep_files)

        # load all files
        print 'Loading input files...',
        for text in dep:
            dep[text] = read_deptree_file(dep_folder + '/' + text + file_suffix_dep)
        print 'done.'

        test(dep, args.model[0], out_folder)


if __name__ == "__main__":
    main()
