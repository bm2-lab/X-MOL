# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import unicodedata
import six
from six.moves import range
import sentencepiece as spm

SPIECE_UNDERLINE = u"▁".encode("utf-8")

def clean_text(text):
  """Performs invalid character removal and whitespace cleanup on text."""
  text = text.replace(u"“", u'"')\
          .replace(u'”', u'"')\
          .replace(u'‘', "'")\
          .replace(u'’', u"'")\
          .replace(u'—', u'-')

  output = []
  for char in text:
      if _is_control(char):
          continue
      if _is_whitespace(char):
          output.append(" ")
      else:
          output.append(char)
  return "".join(output)



def preprocess_text(inputs, remove_space=True, lower=False):
  """preprocess data by removing extra space and normalize data."""

  outputs = inputs
  if remove_space:
    outputs = " ".join(inputs.strip().split())

  if six.PY2 and isinstance(outputs, str):
    try:
      outputs = six.ensure_text(outputs, "utf-8")
    except UnicodeDecodeError:
      outputs = six.ensure_text(outputs, "latin-1")

  outputs = unicodedata.normalize("NFKD", outputs)
  outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
  if lower:
    outputs = outputs.lower()

  return outputs

def encode_pieces(sp_model, text, return_unicode=True, sample=False):
  """turn sentences into word pieces."""

  # liujiaxiang: add for ernie-albert, mainly consider for “/”/‘/’/— causing too many unk
  text = clean_text(text)

  if six.PY2 and isinstance(text, six.text_type):
    text = six.ensure_binary(text, "utf-8")

  if not sample:
    pieces = sp_model.EncodeAsPieces(text)
  else:
    pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)

  new_pieces = []
  for piece in pieces:
    piece = printable_text(piece)
    if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
      cur_pieces = sp_model.EncodeAsPieces(
          six.ensure_binary(piece[:-1]).replace(SPIECE_UNDERLINE, b""))
      if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
        if len(cur_pieces[0]) == 1:
          cur_pieces = cur_pieces[1:]
        else:
          cur_pieces[0] = cur_pieces[0][1:]
      cur_pieces.append(piece[-1])
      new_pieces.extend(cur_pieces)
    else:
      new_pieces.append(piece)

  # note(zhiliny): convert back to unicode for py2
  if six.PY2 and return_unicode:
    ret_pieces = []
    for piece in new_pieces:
      if isinstance(piece, str):
        piece = six.ensure_text(piece, "utf-8")
      ret_pieces.append(piece)
    new_pieces = ret_pieces

  return new_pieces


def encode_ids(sp_model, text, sample=False):

  pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
  ids = [sp_model.PieceToId(piece) for piece in pieces]
  return ids


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    fin = open(vocab_file)
    for num, line in enumerate(fin):
        items = convert_to_unicode(line.rstrip()).split("\t")
        if len(items) > 2:
            break
        token = items[0]
        index = items[1] if len(items) == 2 else num
        token = token.strip()
        vocab[token] = int(index)
    return vocab


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        if item in vocab:
           term = item
        else:
           term = '[UNK]'
        output.append(vocab[term])
    return output


def convert_tokens_to_ids_include_unk(vocab, tokens, unk_token="[UNK]"):
    output = []
    for token in tokens:
        if token in vocab:
            output.append(vocab[token])
        else:
            output.append(vocab[unk_token])
    return output


def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a peice of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

    def merge_subword(self, tokens):
        ret = []
        for token in tokens:
            if token.startswith("##"):
                real_token = token[2:]
                if len(ret):
                    ret[-1] += real_token
                else:
                    ret.append(real_token)
            else:
                ret.append(token)

        return ret

class MolTokenizer(object):
    """Runs end-to-end tokenziation."""
    def __init__(self, vocab_file, do_lower_case=False):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        #self.tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, smi):
        tokens = []
        dc_a = ('l','r')
        marker = 0 # 1,2 indicates that the last token is not complete
        for c in smi:
            if marker == 0:
                if c in dc_a:
                    tokens[-1] += c
                else:
                    tokens.append(c)
                    if c == '[':
                        marker = 1
                    elif c == '%':
                        marker = 2
                        marker_l = 2 #indicates that remain length of %**
            else:
                tokens[-1] += c
                if marker == 1:
                    if c == ']':
                        marker = 0
                elif marker == 2:
                    marker_l -= 1
                    if marker_l == 0:
                        marker = 0
        return tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)

    def merge_subword(self, tokens):
        ret = [token for token in tokens]
        return ret

  

class CharTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in text.lower().split(" "):
            for sub_token in self.tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
            do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
        (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def tokenize_chinese_chars(text):
    """Adds whitespace around any CJK character."""

    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp):
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output
