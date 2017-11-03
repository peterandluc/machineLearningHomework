# coding: utf-8
#
# Hot Topics in Machine Learning, University of Mannheim, 2017
# Author: Rainer Gemulla
# Assignment 3


class Dist:
    """A distribution over a set of categorical variables.

    If values is `None`, all values are set to 0.

    Members
    -------
    vars : list of ``Var``
        Variables used in this distribution. Order matters.
    values : ndarray
        Probabilities (or other arbitrary real values) associated with each
        variable assignment. Each dimension corresponds to a variable, in the
        order of `vars`. The size of each dimension is given by the size of the
        domain of its corresponding variable. Each entry is the probability for
        the particular assignment (i.e., each variable takes the value specified
        by the index in its dimension).
    """
    def __init__(self, vars, values=None):
        self.vars = vars
        self.values = np.zeros([v.K for v in vars]) if values is None else values

    def __str__(self):
        return disttable(self, 'value')

    def value(self):
        """Return the entry associated with the current values of all variables."""
        return self.values[tuple([var.value for var in self.vars])]

    def names(self):
        """Return names of the variables in this distribution.."""
        return [var.name for var in self.vars]

    def values(self):
        """Return values of all variables as a list."""
        return [ var.value for var in self.vars ]

    def indexOf(self, var):
        """Index of the given ``var`` in this distribution.

        Returns `None` if given variable is not in this distributions variables.
        """
        for i in range(len(self.vars)):
            if self.vars[i] == var:
                return i
        return None

    def sample(self, n):
        """Return `n` independent samples from this distribution.

        This method assumes that ``values`` are non-negative and sum to unity,
        i.e., that the distribution is normalized.

        Parameters
        ----------
        n : int
            number of samples to take

        Returns
        -------
        `Dist` containing the sample

        The returned `Dist` has the variables as this distribution. Each entry,
        however, corresponds to the number of times the corresponding value has
        been sampled.
        """
        s = numpy.random.multinomial(n, self.values.ravel())
        s.shape = self.values.shape
        return Dist(self.vars, s)

    def normalize(self):
        """Returns a normalized copy of this distribution.

        Rescales all values such that they sum to 1. This method assumes that
        all values are non-negative reals and that there is at least one
        non-zero value.
        """
        return Dist(self.vars, self.values/np.sum(self.values))

    def marginalize(self, vars):
        """Return a copy of distribution in which `vars` have been marginalized out."""
        if not isinstance(vars, list):
            vars = [ vars ]
        result = self.copy()
        for var in vars:
            i = result.indexOf(var)
            result.values = np.sum(result.values, i)
            del result.vars[i]
        return result

    def fix(self, var, value):
        """Return a copy of this distribution in which `var` has been fixed to `value`."""
        i = self.indexOf(var)
        new_values = self.values.take(value, i)
        new_vars = self.vars.copy()
        del new_vars[i]
        return Dist(new_vars, new_values)

    def copy(self):
        """Returns a copy of this distribution."""
        return Dist(self.vars.copy(), self.values.copy())

    def asfactor(self):
        return Factor(self.vars, self.values)

def disttable(dists, labels=None):
    if not isinstance(dists, list):
        dists = [ dists ]
        if labels is not None:
            labels = [ labels ]
    if labels is None:
        if len(dists)==1:
            labels = ['p']
        else:
            labels = list(map(lambda x:x[0]+str(x[1]),
                              zip(['p']*len(dists), np.arange(len(dists)))))
    table = [ list(i) + list(map(lambda dist: dist.values[i], dists)) \
              for i in np.ndindex(dists[0].values.shape) ]
    names = dists[0].names()
    return tabulate(table, names + labels)



class Var:
    """
    A categorical variable.

    Members
    -------
    name : string
        Name of the variable. Used for sorting, but not to identify the
        variable itself.
    K : int
        Domain of variable is {0,...,K-1}
    value : int
        Current value
    """
    def __init__(self, name, K):
        "Create a new variable with the given `name` and range {0,...,`K`}."
        self.name = name
        self.K = K
        self.value = 0

    def __str__(self):
        return self.name + "=" + str(self.value)

    def __cmp__(self, other):
        c = self.name.__cmp__(other.name)
        if c != 0:
            return c
        else:
            return id(self)-id(other)

    def __lt__(self, other):
        if self.name < other.name:
            return True
        if self.name == other.name:
            return id(self) < id(other)
        return False

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return hash(id(self))



class Factor(Dist):
    """A factor for categorical variables.

    Member variable `vars` describes the scope of the factor, member variable
    `values` its values. See ``Dist`` for a more description of these variables.
    """

    def __init__(self, vars, values=None):
        """
        Create a new factor with the scope `vars`.

        If `values` is none, all values are initialized to 1.
        """
        self.vars = vars
        self.values = np.ones([v.K for v in vars]) if values is None else values

    def __mul__(self, other):
        """Compute factor product.

        The variables in the resulting factor are guaranteed to be sorted.
        """
        ## determine vars of result and how to map result indexes
        vars = sorted(set(self.vars + other.vars))
        indexes_self = np.zeros(len(self.vars), dtype='int')
        indexes_other = np.zeros(len(other.vars), dtype='int')
        for i, var in enumerate(vars):
            i_self = self.indexOf(var)
            if i_self is not None:
                indexes_self[i_self] = i
            i_other = other.indexOf(var)
            if i_other is not None:
                indexes_other[i_other] = i

        ## now create the result
        result = Factor(vars)
        for i in np.ndindex(result.values.shape):
            indexes = np.array(i)
            i1 = tuple(indexes[indexes_self])
            i2 = tuple(indexes[indexes_other])
            result.values[i] = self.values[i1] * other.values[i2]
        return result



class FactorGraph:
    """A factor graph.

    Members
    -------
    vars  : ``SortedSet`` of ``Var``
        The variables in this factor graph
    factors : set of ``Factor``
        The factors in this factor graph
    factors_for : dict from ``Var`` to list of ``Factor``
        The factors connected to each variable
  """
    def __init__(self):
        """Construct a new, empty factors graph."""
        self.vars = SortedSet()
        self.factors = set()
        self.factors_for = dict()

    def add_var(self, var):
        """Add a variable to the factor graphs (if it does not yet exist)."""
        self.vars.add(var)
        if var not in self.factors_for:
            self.factors_for[var] = set()

    def add_vars(self, vars):
        for var in vars:
            self.add_var(var)

    def remove_var(self, var):
        """Remove a variable and all its connected factors from the factor graph."""
        if var in self.vars:
            factors = self.factors_for[var].copy()
            for factor in factors:
                self.remove_factor(factor)
            self.vars.remove(var)
            self.factors_for.pop(var)

    def remove_vars(self, vars):
        for var in vars:
            self.remove_var(var)

    def indexOf(self, var):
        """Index of the given ``var`` in this factor graph.

        Returns `None` if given variable is not in this distributions variables.
        """
        try:
            return self.vars.index(var)
        except ValueError:
            return None

    def add_factor(self, factor):
        """Add a factor to the factor graphs (if it does not yet exist).

        The variables in the scope of this factor must be present in this factor
        graph.
        """
        self.factors.add(factor)
        for var in factor.vars:
            self.factors_for[var].add(factor)

    def add_factors(self, factors):
        for factor in factors:
            self.add_factor(factor)

    def remove_factor(self, factor):
        """Remove a factor from the factor graph (if present)."""
        self.factors.remove(factor)
        for var in factor.vars:
            self.factors_for[var].remove(factor)

    def remove_factors(self, factors):
        for factor in factors:
            self.remove_factor(factor)

    def copy(self):
        """"Copy this factor graph.

        The returned factor graph shares all variables and references with this
        factor graph, but can be independently modified.
        """
        result = FactorGraph()
        result.vars = self.vars.copy()
        result.factors = self.factors.copy()
        result.factors_for = dict()
        for k,v in self.factors_for.items():
            result.factors_for[k] = v.copy()
        return result

    def value(self):
        """Unnormalized probability of the current variable assignment."""
        return np.prod([ factor.value() for factor in self.factors])

    def names(self):
        """Return names of the variables in this factor graph."""
        return [var.name for var in self.vars]

    def values(self):
        """Return values of all variables as a list."""
        return [ var.value for var in self.vars ]

    def __str__(self):
        return "FactorGraph ({} variables, {} factors)".format(len(self.vars), len(self.factors))



def run_gibbs(G, vars, n, warmup=0, skip=0, marginals=False, normalize=False, log=False):
    """Performs Gibbs sampling and returns aggregate statistics of the result.

    Resamples each variable in `vars` from factor graph `G` (= a Gibbs sampling
    pass). Repeats `n` times.

    If `marginals` is `False`, returns the observed frequency of each possible
    assignment of `vars` across all samples as a ``Dist``. Otherwise, returns a
    list with one entry per variable in `vars`, holding that variables observed
    frequencies in the samples.

    Runs `warmup` Gibbs sampling passes before taking the first sample. Skips
    `skip` Gibbs sampling passes before taking each next one. The total number
    of sampling passes is `warmup`+`n`+(`n`-1)*`skip`.

    If `normalize` is true, returns relative instead of absolute frequencies.

    Argument `log` is passed to the underlying ``gibbs`` method.
    """
    if marginals:
        counts = [ np.zeros(var.K) for var in vars ]
    else:
        counts = Dist(vars)
    for i in range(warmup):
        gibbs(G, vars, log)
    for i in range(n):
        gibbs(G, vars, log)
        var_values = [ var.value for var in vars ]
        if marginals:
            for j in range(len(vars)):
                counts[j][vars[j].value] += 1
        else:
            counts.values[ tuple(var_values) ] += 1
        if i<n-1:
            for s in range(skip):
                gibbs(G, vars, log)
    if normalize:
        if marginals:
            counts = list(map(lambda x: x/np.sum(x), counts))
        else:
            counts.values = counts.values/n

    return counts


def nb_to_factorgraph(model):
    priors = np.exp(model['logpriors'])
    cls = np.exp(model['logcls'])
    C, D, K = cls.shape

    # create factor graph and variables
    G = FactorGraph()
    Y = Var("Y", C)
    G.add_var(Y)
    Xs = []
    for j in range(D):
        X = Var("X{:03d}".format(j), K)
        Xs.append(X)
        G.add_var(X)

    # create factor for prior
    G.add_factor( Factor([Y], priors) )

    # create factor for each X
    for j in range(D):
        G.add_factor( Factor([Y,Xs[j]], cls[:,j,:]) )

    # all done
    return G, Y, Xs

def logsumexp(x):
    """Computes log(sum(exp(x)).

    Uses offset trick to reduce risk of numeric over- or underflow. When x is a
    1D ndarray, computes logsumexp of its entries. When x is a 2D ndarray,
    computes logsumexp of each row.

    Keyword arguments:
    x : a 1D or 2D ndarray
    """
    offset = np.max(x, axis=0)
    return offset + np.log(np.sum(np.exp(x-offset), axis=0))

def showdigit(x):
    "Show one digit as a gray-scale image."
    plt.imshow(x.reshape(28,28), norm=mpl.colors.Normalize(0,255), cmap='gray')



## to run this:
## - conda install nltk beautifulsoup4
## - in shell:
##   import nltk
##   nltk.downlad()
## - select Models/average_perceptron_tagger for download
## - in shell:
##   from bs4 import BeautifulSoup as bs
##   from bs4.element import Tag
## - docs, sentences = load_reuters()
def load_reuters():
    with codecs.open("data/reuters.xml", "r", "utf-8") as infile:
        soup = bs(infile, "html5lib")

    ## load documents and sentences
    docs = []
    for elem in soup.find_all("document"):
        doc = []
        sentence = []

        # Loop through each child of the element under "textwithnamedentities"
        for c in elem.find("textwithnamedentities").children:
            if type(c) == Tag:
                if c.name == "namedentityintext":
                    label = 1  # part of a named entity
                else:
                    label = 0  # irrelevant word
                # naive tokenization and sentence boundary detection
                for token in c.text.split():
                    if len(token)>0:
                        if label==0 and token.endswith("."):
                            if len(token)>1:
                                sentence.append((token[0:-1], label))
                            doc.append(sentence)
                            sentence = []
                        else:
                            sentence.append((token, label))
        if len(sentence)>0:
            doc.append(sentence)
        docs.append(doc)

    ## flatten and add part-of-speech tags
    sentences = list(itertools.chain.from_iterable(docs))
    sentences_pos = []
    for i, sentence in enumerate(sentences):
        tokens = [token for token,label in sentence]
        tags = nltk.pos_tag(tokens)
        sentences_pos.append( [(token,pos_tag,label)
                               for (token, label), (token, pos_tag) in zip(sentence, tags)] )

    return docs, sentences_pos

def crf_to_factorgraph(f, info):
    G = FactorGraph()
    n = len(f)

    # create variables
    Ys = []
    for j in range(n):
        Yj = Var('Y{:03d}'.format(j), 2)
        Ys.append(Yj)
    G.add_vars(Ys)

    # create transition factors
    values = np.zeros((2,2))
    for l_from in range(2):
        for l_to in range(2):
            values[l_from,l_to] = info.transitions[ (str(l_from), str(l_to)) ]
    for j in range(n-1):
        G.add_factor(Factor([Ys[j], Ys[j+1]], np.exp(values)))

    # create state factors
    for j in range(n):
        for feature in f[j]:
            values = np.zeros(2)
            values[0] = info.state_features.get((feature, '0'), 0)
            values[1] = info.state_features.get((feature, '1'), 0)
            G.add_factor(Factor([Ys[j]], np.exp(values)))

    # all done, return factor graph
    return G
