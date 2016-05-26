import time
import sys, getopt
import numpy as NP
import theano
#theano.config.allow_gc=False
import theano.tensor as T
import theano.tensor.nnet as NN
from wrapper import *
from txt_data_word import *

def regular_loss(W_list, batch_size):
    cost = 0.
    for i in W_list:
        cost = cost + REGULAR_FACTOR * 0.5 * T.sum(i**2) / batch_size
    return cost

class Moving_AVG(object):
	def __init__(self, array_size=500):
		self.array_size=array_size
		self.queue = NP.zeros((array_size, ))
		self.idx = 0
		self.filled = False
	def append(self, x):
		self.queue[self.idx]=x
		self.idx = (self.idx + 1)%self.array_size
		if not self.filled and self.idx==0:
			self.filled = True
	def get_avg(self):
		result = NP.mean(self.queue) if self.filled else NP.sum(self.queue)/NP.sum(self.queue!=0)
		return result

try:
    opts, args = getopt.getopt(sys.argv[1:], "h", ["ptb", "bbc", "imdb", "wiki", "de", "cs", "train", "test", "baseline", "context_free", 'context_dependent'])
except getopt.GetoptError:
    print 'Usage, please type --ptb, --bbc, --imdb, --wiki, to determine which dataset, --train or --test, and the model can be --baseline, --context_free, --context_dependent'
    sys.exit(2)


fname = './saves/mscale'

gru1_dim = 200 # gru in character LM 
gru2_dim = 400 # gru in word LM
emb_dim = 200 # word emb dim, must equal to gru1_dim due to the high way network
char_emb_dim = 15
train_batch_size = 128
test_batch_size = 256
drop_flag = False

train_flag = False
mode = None
for opt, arg in opts:
    if opt == '-h':
        print 'Usage, please type --ptb, --bbc, --imdb, --wiki, to determine which dataset, --train or --test to select training (without validation) or testing (testing on validation set), the model type was --baseline, --context_free, --context_dependent'
        sys.exit(2)
    if opt == '--ptb':
        fname = fname + '_ptb'
        dict_fname = 'ptb_dict'
        train_fname = 'data/ptb_train.txt'
        test_fname = 'data/ptb_valid.txt'
        DICT_SIZE = 10000
        dataset_fnames = ['data/ptb_train.txt', 'data/ptb_valid.txt', 'data/ptb_test.txt']
        raw_file = True
    if opt == '--bbc':
        fname = fname + '_bbc'
        dict_fname = 'bbc_dict'
        train_fname = 'BBC_train.txt'
        test_fname = 'BBC_valid.txt'
        DICT_SIZE = 10000
        dataset_fnames = ['BBC_train.txt', 'BBC_valid.txt', 'BBC_test.txt']
        raw_file = False
    if opt == '--imdb':
        fname = fname + '_imdb'
        dict_fname = 'imdb_dict'
        train_fname = 'data/imdb_train.txt'
        test_fname = 'data/imdb_valid.txt'
        DICT_SIZE = 30000
        dataset_fnames = ['data/imdb_train.txt', 'data/imdb_valid.txt', 'data/imdb_test.txt']
        raw_file = False
    if opt == '--cs':
        fname = fname + '_cs'
        dict_fname = 'cs_dict'
        train_fname = 'data/cs.train'
        test_fname = 'data/cs.valid'
        DICT_SIZE = 30000
        dataset_fnames = ['data/cs.train', 'data/cs.valid', 'data/cs.test']
        raw_file = False
    if opt == '--de':
        fname = fname + '_de'
        dict_fname = 'de_dict'
        train_fname = 'data/de.train'
        test_fname = 'data/de.valid'
        DICT_SIZE = 30000
        dataset_fnames = ['data/de.train', 'data/de.valid', 'data/de.test']
        raw_file = False
    if opt == '--wiki':
        fname = fname + '_wiki'
        dict_fname = 'wiki_dict'
        train_fname = 'wiki_train.txt'
        test_fname = 'wiki_valid.txt'
        DICT_SIZE = 30000
        dataset_fnames = ['wiki_train.txt', 'wiki_valid.txt', 'wiki_test.txt']
        raw_file = True
    if opt == '--train':
        train_flag = True
    if opt == '--test':
        train_flag = False
    if opt == '--baseline':
        mode = 'baseline'
    if opt == '--context_free':
        mode = 'context_free'
    if opt == '--context_dependent':
        mode = 'context_dependent'

fname = fname + '_' + mode
print 'File name = ', fname
Dataset = txt_data(train_fname, test_fname, dict_fname, dataset_fnames, gen_dict=False, only_dict=True, DICT_SIZE = DICT_SIZE, raw_file = raw_file)
in_dim = Dataset.c_dim # character dict size
out_dim = Dataset.c_dim 
word_out_dim = Dataset.w_dim # word dict size

var_lr = theano.shared(NP.asarray(1.0, dtype=theano.config.floatX)) #learning rate of rmsprop
char_in = T.imatrix()
cw_index1 = T.imatrix() #convert word embedding matrix from sparse to dense. using to speed up 
cw_index2 = T.imatrix()
word_target = T.imatrix()
model = Model()
#model.load('./saves/mscale_de_context_free_34')
drop2 = Dropout(shape=(word_target.shape[0], emb_dim), prob=0.1)
drop3 = Dropout(shape=(word_target.shape[0], gru2_dim), prob=0.1)
drop4 = Dropout(shape=(word_target.shape[0], emb_dim), prob=0.1)

def categorical_crossentropy(prob, true_idx):
    true_idx = T.arange(true_idx.shape[0]) * word_out_dim + true_idx
    t1 = prob.flatten()[true_idx]
    return -T.log(t1)

#NULL a b c I J    input
#a    b c I J K    output
#0    0 0 1 0 0 1  mask

#get mask for handle variable length in batch, all zeros vector will be masked out
def get_mask(cur_in, mask_value=0.):
    return T.shape_padright(T.any((1. - T.eq(cur_in, mask_value)), axis=-1))

#character step

def high_way(cur_in=None, name='', shape=[]):
	g = NN.sigmoid(model.fc(cur_in = cur_in, name=name+'_g', shape=shape))
	h = T.tanh(model.fc(cur_in=cur_in, name=name+'_h', shape=shape))
	return h*g + cur_in * (1. - g)

def _word_step_embedding(word):
        batch_mask = get_mask(word, -1.)
        word_emb = batch_mask * model.embedding(cur_in = word, name='word_embedding', shape=(word_out_dim, emb_dim))
        return word_emb

def _char_step_context_free(char, prev_h1):
	batch_mask = get_mask(char, -1.)
        char = model.embedding(cur_in = char, name = 'char_emb', shape = (in_dim, char_emb_dim))
	gru1 = batch_mask * model.gru(cur_in = char, rec_in = prev_h1, name = 'gru_char', shape = (char_emb_dim, gru1_dim))
	return gru1

def _gen_word_emb_step(h_context_free, h_before):
        batch_mask = get_mask(h_context_free)
	D_h_c = h_before 
	D_h_cf = h_context_free
        D_h = 0.5 * (D_h_c + D_h_cf)
        hw_h = high_way(cur_in = D_h, name = 'hw_emb_h', shape=(emb_dim, emb_dim))
	#hw_c = high_way(cur_in = D_h_c, name = 'hw_emb_c', shape=(emb_dim ,emb_dim))
        #hw_c = high_way(cur_in = hw_c, name = 'hw_emb_c2', shape=(emb_dim, emb_dim))
	#hw_cf = high_way(cur_in = D_h_cf, name = 'hw_emb_cf', shape=(gru1_dim, emb_dim))
        #hw_cf = high_way(cur_in = hw_cf, name = 'hw_emb_cf2', shape=(emb_dim, emb_dim))
	#word_emb = batch_mask * 0.5 * (hw_c+hw_cf)
        word_emb = batch_mask * hw_h
	#gru_emb = model.gru(cur_in = word_emb, rec_in = h_before, name = 'gru_emb', shape = (emb_dim, emb_dim))
	#return gru_emb, word_emb
	h_before = h_before * 0.5 + word_emb * 0.5
	return h_before, word_emb

def _gen_word_emb_step_free(h_context_free):
        batch_mask = get_mask(h_context_free) 
	D_h_cf = h_context_free
	hw_cf = high_way(cur_in = D_h_cf, name = 'hw_emb_cf', shape=(gru1_dim, emb_dim))
        #hw_cf = high_way(cur_in = hw_cf, name = 'hw_emb_cf2', shape=(emb_dim, emb_dim))
	word_emb = batch_mask * hw_cf
	return word_emb

def softmax(x):
    e_x = T.exp(x)
    sm = e_x / e_x.sum(axis=1, keepdims=True)
    return sm

#import theano.sandbox.cuda.dnn as CUDNN
#dnnsoftmax =  CUDNN.GpuDnnSoftmax('bc01', 'fast', 'channel')
#def softmax(x):
#    ret =dnnsoftmax(x.dimshuffle(0, 1, 'x', 'x'))
#    return ret[:, :, 0, 0]

def word_step(word_emb, prev_h1):
	batch_mask = get_mask(word_emb)
        if drop_flag:
            D_word_emb = drop2.drop(word_emb)
        else:
            D_word_emb = word_emb

	gru1 = model.gru(cur_in = D_word_emb, rec_in = prev_h1, name = 'gru_word', shape = (emb_dim, gru2_dim))

	if drop_flag:
            D_gru1 = drop3.drop(gru1)
	else:
            D_gru1 = gru1

	return gru1, D_gru1

drop_flag = False
EPSI = 1e-15
def get_express(train=False, emb_flag=None):
	global drop_flag
	drop_flag = train
	batch_size = word_target.shape[0]

        word_embs=None
        if emb_flag=='context_free' or emb_flag=='context_dependent':
            sc, _ = theano.scan(_char_step_context_free, sequences=[char_in.dimshuffle(1,0)], outputs_info = [T.zeros((char_in.shape[0], gru1_dim))], name='scan_char_rnn', profile=False)

            # assign character time step to word time step
            h_context_free = sc.dimshuffle(1,0,2)
	    h_context_free = h_context_free.reshape((sc.shape[0] * sc.shape[1], gru1_dim))
    	    cw_index = cw_index1 * sc.shape[0]+ cw_index2
	    h_context_free = h_context_free[cw_index].reshape((cw_index.shape[0], cw_index.shape[1], gru1_dim))
	    h_context_free = h_context_free.dimshuffle(1,0,2)
        
            if emb_flag=='context_dependent':
                sc, _ = theano.scan(_gen_word_emb_step, sequences=[h_context_free], outputs_info = [T.zeros((batch_size, emb_dim)), None], name='scan_gen_emb', profile=False)
                word_embs = sc[1]
            if emb_flag=='context_free':
                sc, _ = theano.scan(_gen_word_emb_step_free, sequences=[h_context_free], name='scan_gen_emb_free')
                word_embs = sc
        if emb_flag=='baseline':
            sc, _ = theano.scan(_word_step_embedding, sequences=[word_target[:, :-1].dimshuffle(1,0)], name='scan_word_emb')
            word_embs = sc

	sc,_ = theano.scan(word_step, sequences=[word_embs], outputs_info = [T.zeros((batch_size, gru2_dim)), None], name='scan_word_rnn', profile=False)

	word_out = sc[1].dimshuffle(1,0,2).reshape((word_target.shape[0]*(word_target.shape[1]-1), gru2_dim))
	word_out = softmax(model.fc(cur_in = word_out, name = 'fc_word', shape=(gru2_dim, word_out_dim)))
	word_out = T.clip(word_out, EPSI, 1.0-EPSI)

	f_word_target = word_target[:,1:].reshape((word_target.shape[0]*(word_target.shape[1]-1), ))
	PPL_word_LM = T.sum((1. -T.eq(f_word_target, -1)) * categorical_crossentropy(word_out, f_word_target))
	cost_word_LM = PPL_word_LM/T.sum(f_word_target>=0)
	cost_all = cost_word_LM

	if train:
                grad_all = rmsprop(cost_all, model.weightsPack.getW_list(), lr=var_lr,epsilon=var_lr**2, rescale = 5. , ignore_input_disconnect=True)
		return cost_all, PPL_word_LM, grad_all
	else:
		return cost_all, PPL_word_LM
	
cost_all, PPL = get_express(train=False, emb_flag=mode)
if mode=='baseline':
    test_func_hw = theano.function([word_target], [cost_all, PPL], allow_input_downcast=True)
if mode=='context_free' or mode=='context_dependent':
    test_func_hw = theano.function([char_in, cw_index1, cw_index2, word_target], [cost_all, PPL], allow_input_downcast=True)

print 'TEST COMPILE'

if train_flag :
    cost_all, PPL, grad_all = get_express(train=True, emb_flag=mode)
    if mode=='baseline':
        train_func_hw = theano.function([word_target], [cost_all, PPL], updates=grad_all, allow_input_downcast=True)
    if mode=='context_free' or mode=='context_dependent':
        train_func_hw = theano.function([char_in, cw_index1, cw_index2, word_target], [cost_all, PPL], updates=grad_all, allow_input_downcast=True)
    print 'TRAIN COMPILE'       

train_func = None
test_func = None

last_valid_PPL = 2e10
valid_increase_cnt = 0 #if loss on validation set didn't decrease in N steps, change the learning rate as half
for i in xrange(600):
    train_func = train_func_hw if train_flag else None
    test_func = test_func_hw

    ma_cost = Moving_AVG(500)
    mytime = time.time()
    if train_flag:
        train_batchs = Dataset.train_size/train_batch_size
        #train_batchs = min(2000, train_batchs)  
        for j in xrange(train_batchs):
            if mode=='baseline':
                word_label = Dataset.get_batch(train_batch_size, only_word=True)
                n_cost, n_ppl = train_func(word_label)
                ma_cost.append(n_cost)
                print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' PPL = ', NP.exp(n_ppl/NP.sum(word_label[:,1:]>=0)), ' AVG Cost = ', ma_cost.get_avg(), 'LEN = ', NP.shape(word_label)[1]

            if mode=='context_free' or mode=='context_dependent':
                char, char_label, word_index1, word_index2, word_label = Dataset.get_batch(train_batch_size)
                n_cost, n_ppl = train_func(char, word_index1, word_index2, word_label)
                ma_cost.append(n_cost)
                print 'Epoch = ', str(i), ' Batch = ', str(j), ' Cost = ', n_cost, ' PPL = ', NP.exp(n_ppl/NP.sum(word_label[:,1:]>=0)), ' AVG Cost = ', ma_cost.get_avg(), 'LEN = ', NP.shape(char)[1], NP.shape(word_index1)[1]

#        print train_func.profile.summary()
	newtime = time.time()
	print 'One Epoch Time = ', newtime-mytime
	mytime = newtime
        model.save(fname+'_'+str(i))
    if not train_flag:
        model.load(fname+'_'+str(i))
    Dataset.test_data_idx = 0
    Dataset.test_len_idx = 0
    test_wper = []
    test_wcnt = []
    test_batchs = Dataset.test_size/test_batch_size
#    test_batchs = min(test_batchs, 50) if train_flag else test_batchs
    for j in xrange(test_batchs):
        if mode=='baseline':
            word_label = Dataset.get_batch(test_batch_size, test=True, only_word=True)
            n_cost, n_ppl = test_func(word_label)
        if mode=='context_free' or mode=='context_dependent':
            char, char_label, word_index1, word_index2, word_label = Dataset.get_batch(test_batch_size, test=True)
            n_cost, n_ppl = test_func(char, word_index1, word_index2, word_label)
        test_wper.append(n_ppl)
        test_wcnt.append(NP.sum(word_label[:,1:]>=0))
        print ' Test Batch = ', str(j), 
    valid_PPL = NP.exp(NP.sum(test_wper)/NP.sum(test_wcnt))
    print '\nEpoch = ', str(i), ' Test Word PPL = ', valid_PPL
    if valid_PPL > last_valid_PPL:
        valid_increase_cnt += 1
        if valid_increase_cnt>=1:
            print 'change learning rate', var_lr.get_value()*0.5
            var_lr.set_value(var_lr.get_value()*0.5)
            valid_increase_cnt = 0
    last_valid_PPL = valid_PPL



