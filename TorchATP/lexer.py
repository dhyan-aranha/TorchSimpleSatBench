import unittest
import string
import re

class Ident:
    """
    Class for generating distinct named objects.
    """

    def __init__ (self, name=""):
        self.name = name

    def __repr__(self):
        return self.name

class ScannerError(Exception):
    """
    A class representing all errors that the scanner can produce.
    """
    def __init__(self):
        self.name = "ScannerError"
        self.value = "<none>"

    def __repr__(self):
        return self.name+"("+repr(self.value)+")"

    def __str__(self):
        return self.__repr__()


class IllegalCharacterError(ScannerError):
    """
    Class representing an unexpexted character error
    """
    def __init__(self, char):
        self.name  = "Illegal character"
        self.value = char


class UnexpectedTokenError(ScannerError):
    """
    Class representing an unexpected token error.
    """
    def __init__(self, token):
        self.name  = "Unexpected token"
        self.value = token



class UnexpectedIdentError(ScannerError):
    """
    Class representing an unexpected identifier error.
    """
    def __init__(self, token):
        self.name  = "Unexpected identifier"
        self.value = token




nl_re  = re.compile(r"\n")



class Token(object):
    """
    Represent a single token with name, position, and print
    representation.
    """
    NoToken        = Ident("No Token")
    WhiteSpace     = Ident("White Space")
    Comment        = Ident("Comment")
    IdentUpper     = Ident("Identifier starting with capital letter")
    IdentLower     = Ident("Identifier starting with lower case letter")
    DefFunctor     = Ident("Defined symbol (starting with a $)")
    Integer        = Ident("Positive or negative Integer")
    FullStop       = Ident(". (full stop)")
    OpenPar        = Ident("(")
    ClosePar       = Ident(")")
    OpenSquare     = Ident("[")
    CloseSquare    = Ident("]")
    Comma          = Ident(",")
    Colon          = Ident(":")
    EqualSign      = Ident("=")
    NotEqualSign   = Ident("!=")
    Nand           = Ident("~&")
    Nor            = Ident("~|")
    Or             = Ident("|")
    And            = Ident("&")
    Implies        = Ident("=>")
    AltImplies     = Ident("->")
    BImplies       = Ident("<=")
    Equiv          = Ident("<=>")
    Xor            = Ident("<~>")
    Universal      = Ident("!")
    Existential    = Ident("?")
    Negation       = Ident("~")
    SQString       = Ident("String in 'single quotes'")
    EOFToken       = Ident("*EOF*")

    def __init__(self, type, literal, source, pos):
        self.type    = type
        self.literal = literal
        self.source  = source
        self.pos     = pos

    def __repr__(self):
        return repr( (self.type, self.literal) )

    def linepos(self):
        """
        Return the line number of the token by counting all the
        newlines in the position up to the current token.
        """
        return len(nl_re.findall(self.source[:self.pos]))+1


class Lexer(object):
    """
    Lexical analysier. This will convert a string into a sequence of
    tokens that can be inspected and processed in-order. It is a bit
    of an overkill for the simple application, but makes actual
    parsing later much easier and more robust than a quicker hack.
    """

    # This list is traversed in order, the first match is
    # returned. This makes it much easier than "longest match", and
    # I have not yet seen a grammar where this causes trouble.
    token_defs = [
        (re.compile(r"\."),                    Token.FullStop),
        (re.compile(r"\("),                    Token.OpenPar),
        (re.compile(r"\)"),                    Token.ClosePar),
        (re.compile(r"\["),                    Token.OpenSquare),
        (re.compile(r"\]"),                    Token.CloseSquare),
        (re.compile(r","),                     Token.Comma),
        (re.compile(r":"),                     Token.Colon),
        (re.compile(r"~\|"),                   Token.Nor),
        (re.compile(r"~&"),                    Token.Nand),
        (re.compile(r"\|"),                    Token.Or),
        (re.compile(r"&"),                     Token.And),
        (re.compile(r"=>"),                    Token.Implies),
        (re.compile(r"->"),                    Token.Implies),
        (re.compile(r"<=>"),                   Token.Equiv),
        (re.compile(r"<="),                    Token.BImplies),
        (re.compile(r"<~>"),                   Token.Xor),
        (re.compile(r"="),                     Token.EqualSign),
        (re.compile(r"!="),                    Token.NotEqualSign),
        (re.compile(r"~"),                     Token.Negation),
        (re.compile(r"!"),                     Token.Universal),
        (re.compile(r"\?"),                    Token.Existential),
        (re.compile(r"\s+"),                   Token.WhiteSpace),
        (re.compile(r"[0-9][0-9]*"),           Token.IdentLower),
        (re.compile(r"[a-z][_a-z0-9_A-Z]*"),   Token.IdentLower),
        (re.compile(r"[_A-Z][_a-z0-9_A-Z]*"),  Token.IdentUpper),
        (re.compile(r"\$[_a-z0-9_A-Z]*"),      Token.DefFunctor),
        (re.compile(r"#[^\n]*"),               Token.Comment),
        (re.compile(r"%[^\n]*"),               Token.Comment),
        (re.compile(r"'[^']*'"),               Token.SQString)
        ]

    def __init__(self, source, name="user string"):
        """
        Initialize the lexer with the string (=sequence of bytes) to
        be split into tokens. The second argument can be used to
        denote the source of the data, e.g. a filename.
        """
        self.token_stack = []
        self.source = source
        self.pos = 0
        self.name = name
       

    def getName(self):
        return self.name
        
    def Push(self, token):
        """
        Return a token to the token stack. This allows basically
        unlimited look-ahead under user control.
        """
        self.token_stack.append(token)

    def Look(self):
        """
        Return the next token without consuming it.
        """
        res = self.Next()
        self.Push(res)
        return res

    def LookLit(self):
        """
        Return the literal value of the next token, i.e. the string
        generating the token.
        """
        return self.Look().literal

    def TestTok(self, tokens):
        """
        Take a list of expected token types. Return True if the
        next token is expected, False otherwise.
        """
        try:
            # If tokens is a list, we accept all elements from the
            # list.
            return self.Look().type in tokens
        except TypeError:
            # Otherwise, it is a single token whose type has to be
            # matched.
            return self.Look().type == tokens

    def CheckTok(self, tokens):
        """
        Take a list of expected token types. If the next token is
        not among the expected ones, exit with an error. Otherwise do
        nothing.
        """
        if not self.TestTok(tokens):
            raise UnexpectedTokenError(
                repr(self.Look().literal)+
                " not "+repr(tokens))

    def AcceptTok(self, tokens):
        """
        Take a list of expected token types. If the next token is
        among the expected ones, consume and return it.
        Otherwise, exit with an error.
        """
        self.CheckTok(tokens)
        return self.Next()


    def TestLit(self, litvals):
        """
        Take a list of expected literal strings. Return True if the
        next token's string value is among them, False otherwise.
        """
        if type(litvals) == type([]):
            return self.LookLit() in litvals
        else:
            return self.LookLit() == litvals

    def CheckLit(self, litvals):
        """
        Take a list of expected literal strings. If the next token's
        literal is not among the expected ones, exit with an
        error. Otherwise do nothing.
        """
        if not self.TestLit(litvals):
            raise UnexpectedIdentError(
                repr(self.Look().literal)+
                " not "+repr(litvals))

    def AcceptLit(self, litvals):
        """
        Take a list of expected literal strings. If the next token's
        literal is among the expected ones, consume and return the
        literal. Otherwise, exit with an error.
        """
        self.CheckLit(litvals)
        return self.Next()


    def Next(self):
        """
        Return next semantically relevant token.
        """
        res = self.NextUnfiltered();
        while res.type in [Token.WhiteSpace, Token.Comment]:
            res = self.NextUnfiltered()
        return res

    def NextUnfiltered(self):
        """
        Return next token, including tokens ignored by most
        languages.
        """
        if len(self.token_stack) > 0:
            return self.token_stack.pop()
        else:
            old_pos = self.pos
            if self.source[old_pos:] == "":
                return Token(Token.EOFToken, "", self.source, old_pos)
            for i in self.token_defs:
                # Go through all the token definitions and process the
                # first one that matchs.
                mr = i[0].match(self.source, self.pos)
                if mr:
                    literal = self.source[mr.start():mr.end()]
                    self.pos = mr.end()
                    type = i[1]
                    break
            if not mr:
                raise IllegalCharacterError(self.source[self.pos:self.pos+4]+"...")

            return Token(type, literal, self.source, old_pos)

