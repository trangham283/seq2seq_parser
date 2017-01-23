#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: set ts=2 sw=2 noet:

import sys
from nlp_util import pstree, render_tree, nlp_eval, treebanks, relaxed_parse_errors

def mprint(text, out_dict, out_name):
	all_stdout = True
	for key in out_dict:
		if out_dict[key] != sys.stdout:
			all_stdout = False
	
	if all_stdout:
		print text
	elif out_name == 'all':
		for key in out_dict:
			print >> out_dict[key], text
	else:
		print >> out_dict[out_name], text


if __name__ == '__main__':
	if len(sys.argv) != 4:
		print "Print parseval scores sentence by sentence"
		print "   %s <gold> <test> <output_prefix>" % sys.argv[0]
		print "Running doctest"
		import doctest
		doctest.testmod()
		sys.exit(0)

	out = {
		'err': sys.stdout,
		'out': sys.stdout,
	}
	if len(sys.argv) > 3:
		prefix = sys.argv[3]
		for key in out:
			out[key] = open(prefix + '.' + key, 'w')
	gold_in = open(sys.argv[1])
	test_in = open(sys.argv[2])
	sent_no = 0
	stats = {
		'out_evalb': [0, 0, 0],
		'out_relaxed': [0, 0, 0]
	}
	

	while True:
		sent_no += 1
		gold_text = gold_in.readline()
		test_text = test_in.readline()
		if gold_text == '' and test_text == '':
			mprint("End of both input files", out, 'err')
			break
		elif gold_text == '':
			mprint("End of gold input", out, 'err')
			break
		elif test_text == '':
			mprint("End of test input", out, 'err')
			break

		mprint("Sentence %d:" % sent_no, out, 'all')

		gold_text = gold_text.strip()
		test_text = test_text.strip()
		if len(gold_text) == 0:
			mprint("No gold tree", out, 'all')
			continue
		elif len(test_text) == 0:
			mprint("Not parsed", out, 'all')
			continue

		gold_complete_tree = pstree.tree_from_text(gold_text,
				allow_empty_labels=True)
		gold_complete_tree = treebanks.homogenise_tree(gold_complete_tree)
		treebanks.ptb_cleaning(gold_complete_tree)
		gold_tree = treebanks.apply_collins_rules(gold_complete_tree, False)
		#gold_tree = gold_complete_tree 
		if gold_tree is None:
			mprint("Empty gold tree", out, 'all')
			mprint(gold_complete_tree.__repr__(), out, 'all')
			mprint(gold_tree.__repr__(), out, 'all')
			continue

		if '()' in test_text:
			mprint("() test tree", out, 'all')
			continue
		test_complete_tree = pstree.tree_from_text(test_text,
				allow_empty_labels=True)
		test_complete_tree = treebanks.homogenise_tree(test_complete_tree)
		treebanks.ptb_cleaning(test_complete_tree)
		test_tree = treebanks.apply_collins_rules(test_complete_tree, False)
		#test_tree = test_complete_tree 
		if test_tree is None:
			mprint("Empty test tree", out, 'all')
			mprint(test_complete_tree.__repr__(), out, 'all')
			mprint(test_tree.__repr__(), out, 'all')
			continue

		gold_words = gold_tree.word_yield()
		test_words = test_tree.word_yield()
		if len(test_words.split()) != len(gold_words.split()):
			mprint("Sentence lengths do not match...", out, 'all')
			mprint("Gold: " + gold_words.__repr__(), out, 'all')
			mprint("Test: " + test_words.__repr__(), out, 'all')

		match_strict, gold_strict, test_strict, _, _ = relaxed_parse_errors.counts_for_prf(test_tree, gold_tree)
		match_relaxed, gold_relaxed, test_relaxed , _, _ = relaxed_parse_errors.relaxed_counts_for_prf(test_tree, gold_tree)
		stats['out_evalb'][0] += match_strict
		stats['out_evalb'][1] += gold_strict
		stats['out_evalb'][2] += test_strict
		p, r, f = nlp_eval.calc_prf(match_strict, gold_strict, test_strict)
		mprint("Eval--Strict Evalb: %.2f  %.2f  %.2f" % (p*100, r*100, f*100), out, 'out')
		stats['out_relaxed'][0] += match_relaxed
		stats['out_relaxed'][1] += gold_relaxed
		stats['out_relaxed'][2] += test_relaxed
		p, r, f = nlp_eval.calc_prf(match_relaxed, gold_relaxed, test_relaxed)
		mprint("Eval--Relaxed Edit: %.2f  %.2f  %.2f" % (p*100, r*100, f*100), out, 'out')

	match = stats['out_evalb'][0]
	gold = stats['out_evalb'][1]
	test = stats['out_evalb'][2]
	p, r, f = nlp_eval.calc_prf(match, gold, test)
	mprint("Overall--Standard EVALB %s: %.2f  %.2f  %.2f" % ('out', p*100, r*100, f*100), out, 'out')
	
	match = stats['out_relaxed'][0]
	gold = stats['out_relaxed'][1]
	test = stats['out_relaxed'][2]
	p, r, f = nlp_eval.calc_prf(match, gold, test)
	mprint("Overall--Relaxed EDIT %s: %.2f  %.2f  %.2f" % ('out', p*100, r*100, f*100), out, 'out')
	
