# Discourse Segmenter

[![The MIT License](https://img.shields.io/dub/l/vibe-d.svg)](http://opensource.org/licenses/MIT)

A collection of various discourse segmenters with pre-trained models for German texts.

## Description

This python module currently comprises two discourse segmenters: *edseg* and *bparseg*.

*edseg* is a rule-based system that uses shallow discourse-oriented parsing to determine boundaries of elementary discourse units in text.  The rules are hard-coded in the [submodule's file](dsegmenter/edseg/clause_segmentation.py) and are only applicable to German input.

*bparseg* is an ML-based segmentation module that operates on syntactic constituency trees (output from [BitPar](http://www.cis.uni-muenchen.de/~schmid/tools/BitPar/)) and decides whether a syntactic constituent initiates a discourse segment or not using a pre-trained linear SVM model.

## Installation

To install this package from the distributed tarball, run
```shell
pip install  https://github.com/WladimirSidorenko/DiscourseSegmenter/archive/0.0.1.dev1.tar.gz
```

Alternatively, you can also install this package directly from source repository by executing:
```shell
git clone git@github.com:WladimirSidorenko/DiscourseSegmenter.git
cd DiscourseSegmenter
./setup.py install
```

## Usage

After installation, you can either import the module in your python scripts (see an example [here](scripts/discourse_segmenter)), e.g.:

```python
from dsegmenter.bparseg import BparSegmenter

segmenter = BparSegmenter()
```

or use the stand-alone script `discourse_segmenter` to process parsed input data, cf.:

```shell
discourse_segmenter --help
```
