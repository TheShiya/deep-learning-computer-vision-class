from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
from . import utils


def log_likelihood(model: LanguageModel, some_text: str):
    """
    Your code here

    Evaluate the log-likelihood of a given string.

    Hint: utils.one_hot might come in handy

    :param model: A LanguageModel
    :param some_text:
    :return: float
    """
    ll_matrix = model.predict_all(some_text)[:,:-1]
    ll = (ll_matrix * utils.one_hot(some_text)).sum()
    return ll.item()


def sample_random(model: LanguageModel, max_length: int = 100):
    """
    Your code here.

    Sample a random sentence from the language model.
    Terminate once you reach a period '.'

    :param model: A LanguageModel
    :param max_length: The maximum sentence length
    :return: A string
    """
    output = ''
    curr_char = ''

    while curr_char != '.' and len(output) < max_length:
        log_probs = model.predict_next(curr_char)
        dist = torch.distributions.categorical.Categorical(logits=log_probs)
        curr_char = utils.vocab[dist.sample([1]).item()]
        output += curr_char

    return output


class TopNHeap:
    """
    A heap that keeps the top N elements around
    h = TopNHeap(2)
    h.add(1)
    h.add(2)
    h.add(3)
    h.add(0)
    print(h.elements)
    > [2,3]

    """
    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, e):
        from heapq import heappush, heapreplace
        if len(self.elements) < self.N:
            heappush(self.elements, e)
        elif self.elements[0] < e:
            heapreplace(self.elements, e)


def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100, average_log_likelihood: bool = False):
    """
    Your code here

    Use beam search for find the highest likelihood generations, such that:
      * No two returned sentences are the same
      * the `log_likelihood` of each returned sentence is as large as possible

    :param model: A LanguageModel
    :param beam_size: The size of the beam in beam search (number of sentences to keep around)
    :param n_results: The number of results to return
    :param max_length: The maximum sentence length
    :param average_log_likelihood: Pick the best beams according to the average log-likelihood, not the sum
                                   This option favors longer strings.
    :return: A list of strings of size n_results
    """
    heap = TopNHeap(N=beam_size)
    vocab = 'abcdefghijklmnopqrstuvwxyz .'

    next_likes = model.predict_next('')
    for i, next_like in enumerate(next_likes):
        element = ((next_like).item(), vocab[i])
        if element not in [e[1] for e in heap.elements]:
            heap.add(element)

    for depth in range(max_length):        
        for like, string in heap.elements:
            if string[-1] == '.':
                continue
            next_likes = model.predict_next(string[-1])
            for i, next_like in enumerate(next_likes):
                if average_log_likelihood:
                    curr_len = len(string)
                    avg_like = (like*curr_len + next_like.item())/(curr_len + 1)
                    element = (avg_like, string + vocab[i])
                else:
                    element = (like + next_like.item(), string + vocab[i])            
                if element[1] not in [e[1] for e in heap.elements]:
                    heap.add(element)

    sorted_heap = sorted(heap.elements, key=lambda x: x[0])[::-1]
    output = [e[1] for e in sorted_heap[:n_results]]
    return output


if __name__ == "__main__":
    """
      Some test code.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Adjacent')
    args = parser.parse_args()

    lm = AdjacentLanguageModel() if args.model == 'Adjacent' else (load_model() if args.model == 'TCN' else Bigram())

    for s in ['abcdefg', 'abcgdef', 'abcbabc', '.abcdef', 'fedcba.']:
        print(s, float(log_likelihood(lm, s)))
    print()

    for i in range(10):
        s = sample_random(lm)
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100):
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100, average_log_likelihood=True):
        print(s, float(log_likelihood(lm, s)) / len(s))
