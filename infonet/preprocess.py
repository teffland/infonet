import json
import io
from glob import glob
import random
from scipy.misc import comb # to check for num relations

from infonet.vocab import Vocab
from infonet.util import convert_sequences

# NOTE: all mapping schemes assume the annotation spans correspond with python slice indexing
# eg ann-span = [0,2] means the interval [0,2) or tokens[0:2]

def Entity_BIO_map(mention_labels, annotation):
    """ Uses BIO scheme (untyped) for entities only """
    if annotation['node-type'] == 'entity':
        left, right = tuple(annotation['ann-span'])
        mention_labels[left] = 'B-entity'
        for i in range(1, right-left):
            mention_labels[left+i] = 'I-entity'
    return mention_labels

def E_BIO_map(mention_labels, annotation):
    """ Uses BIO scheme (typed) for entities only """
    if annotation['node-type'] in ('entity', 'event-anchor'):
        left, right = tuple(annotation['ann-span'])
        mention_labels[left] = 'B-'+annotation['node-type']
        for i in range(1, right-left):
            mention_labels[left+i] = 'I-'+annotation['node-type']
    return mention_labels

def NoVal_BIO_map(mention_labels, annotation):
    """ Uses BIO scheme (typed) for entities only """
    if annotation['node-type'] in ('entity', 'event-anchor'):
        left, right = tuple(annotation['ann-span'])
        mention_labels[left] = 'B'
        for i in range(1, right-left):
            mention_labels[left+i] = 'I'
    return mention_labels

def Entity_typed_BIO_map(mention_labels, annotation):
    """ Uses BIO scheme (typed) for entities only """
    if annotation['node-type'] == 'entity':
        mention_type = annotation['type']
        left, right = tuple(annotation['ann-span'])
        mention_labels[left] = 'B-entity-'+mention_type
        for i in range(1, right-left):
            mention_labels[left+i] = 'I-entity-'+mention_type
    return mention_labels

def All_typed_BIO_map(mention_labels, annotation):
    """ Uses BIO scheme (typed) for entities only """
    mention_type = annotation['type']
    left, right = tuple(annotation['ann-span'])
    mention_labels[left] = 'B-'+annotation['node-type']+'-'+mention_type
    for i in range(1, right-left):
        mention_labels[left+i] = 'I-'+annotation['node-type']+'-'+mention_type
    return mention_labels

def E_typed_BIO_map(mention_labels, annotation):
    """ Uses BIO scheme (typed) for entities and event anchors only """
    if annotation['node-type'] in ('entity', 'event-anchor'):
        if annotation['node-type'] == 'entity':
            mention_type = annotation['type']
        else:
            mention_type = annotation['subtype']
        left, right = tuple(annotation['ann-span'])
        mention_labels[left] = 'B-'+annotation['node-type']+'-'+mention_type
        for i in range(1, right-left):
            mention_labels[left+i] = 'I-'+annotation['node-type']+'-'+mention_type
    return mention_labels

def Entity_BILOU_map(mention_labels, annotation):
    """ Uses BILOU scheme (untyped) for entities only """
    if annotation['node-type'] == 'entity':
        left, right = tuple(annotation['ann-span'])
        if left == (right-1):
            mention_labels[left] = 'U'
        else:
            mention_labels[left] = 'B'
            for i in range(1, right-left-1):
                mention_labels[left+i] = 'I'
            mention_labels[right-1] = 'L'
    return mention_labels

def Entity_typed_BILOU_map(mention_labels, annotation):
    """ Uses BILOU scheme (typed) for entities only """
    if annotation['node-type'] == 'entity':
        mention_type = annotation['type']
        left, right = tuple(annotation['ann-span'])
        if left == (right-1):
            mention_labels[left] = 'U-entity-'+mention_type
        else:
            mention_labels[left] = 'B-entity-'+mention_type
            for i in range(1, right-left-1):
                mention_labels[left+i] = 'I-entity-'+mention_type
            mention_labels[right-1] = 'L-entity-'+mention_type
    return mention_labels

def BIO_map(mention_labels, annotation):
    """ Uses BIO scheme (untyped) for entities only """
    left, right = tuple(annotation['ann-span'])
    mention_labels[left] = 'B'
    for i in range(1, right-left):
        mention_labels[left+i] = 'I'
    return mention_labels

def typed_BIO_map(mention_labels, annotation):
    """ Uses BIO scheme (typed) for entities only """
    mention_type = annotation['node-type']
    left, right = tuple(annotation['ann-span'])
    mention_labels[left] = 'B-'+mention_type
    for i in range(1, right-left):
        mention_labels[left+i] = 'I-'+mention_type
    return mention_labels

def BILOU_map(mention_labels, annotation):
    """ Uses BILOU scheme (untyped) for entities only """
    left, right = tuple(annotation['ann-span'])
    if left == (right-1):
        mention_labels[left] = 'U'
    else:
        mention_labels[left] = 'B'
        for i in range(1, right-left-1):
            mention_labels[left+i] = 'I'
        mention_labels[right-1] = 'L'
    return mention_labels

def typed_BILOU_map(mention_labels, annotation):
    """ Uses BILOU scheme (typed) for entities only """
    mention_type = annotation['node-type']
    left, right = tuple(annotation['ann-span'])
    if left == (right-1):
        mention_labels[left] = 'U-'+mention_type
    else:
        mention_labels[left] = 'B-'+mention_type
        for i in range(1, right-left-1):
            mention_labels[left+i] = 'I-'+mention_type
        mention_labels[right-1] = 'L-'+mention_type
    return mention_labels

def compute_flat_mention_labels(doc, scheme_func=Entity_BIO_map):
    """ Takes a YAAT style document and computes token-level mention label list.

    This function only considers the outermost spans (as per ACE evaluation)
    by editing the mentions of shortest span-length to longest.
    Thus wider mentions will override narrower nested mentions.

    Example:
    --------

    tokens: ['part', 'of' , 'mention', '.', 'not', 'part']
    with annotation: {'ann-type':'node',
                      'node-type':'entity',
                      'ann-span':[0,2]}
    with scheme: BIO

    yields:
    mention_labels = ['B', 'I', 'I', 'O', 'O', 'O']
    """
    mention_labels = ['O' for token in doc['tokens']]
    mentions = [ annotation for annotation in doc['annotations'] if annotation['ann-type'] == 'node' ]
    # create annotations from shortest to longest, so that largest spans overwrite smaller
    annotations = sorted(mentions, key=lambda x:x['ann-span'][1]-x['ann-span'][0])
    for annotation in annotations:
        mention_labels = scheme_func(mention_labels, annotation)
    return mention_labels

def compute_tag_map(boundary_vocab):
    """ Automatically computes the data needed for decoding mentions
    from tags, given the tagset we are using
    """
    tag_map = {
        'start_tags':tuple([ t for t in boundary_vocab.vocabset
                             if t.startswith(('B', 'U'))]),
        'in_tags':tuple([ t for t in boundary_vocab.vocabset
                             if t.startswith(('B', 'U', 'I', 'L'))]),
        'out_tags':tuple([ t for t in boundary_vocab.vocabset
                             if t.startswith('O')]),
        'tag2mtype':{t:'-'.join(t.split('-')[1:])
                    if len(t.split('-')) > 1
                    else '<UNK>'
                    for t in boundary_vocab.vocabset
                    }
    }
    return tag_map

def compute_typemaps(docs):
    """ Compute allowable type constraints from docs """
    mtype2msubtype = {}
    msubtype2rtype = {'left':{}, 'right':{}}
    for doc in docs:
        # eg 'entity' -> {'entity:PER', 'entity:ORG', ...}
        mspan2msubtype = {} # used for relations
        for m in doc['mentions']:
            msubtype = m[2]
            mspan2msubtype[m[:2]] = msubtype
            mtype = msubtype.split(':')[0]
            if mtype in mtype2msubtype:
                mtype2msubtype[mtype] |= {msubtype}
            else:
                mtype2msubtype[mtype] = {msubtype}

        # print {r[4] for r in doc['relations']}
        for r in doc['relations']:
            rtype = r[4]
            left_msubtype = mspan2msubtype[r[:2]]
            right_msubtype = mspan2msubtype[r[2:4]]
            # print left_msubtype, rtype, right_msubtype
            # eg, 'entity:PER' -> ('relation:--ORG-AFF-->', ...) headed forward
            if left_msubtype in msubtype2rtype['left']:
                msubtype2rtype['left'][left_msubtype] |= {rtype}
            else:
                msubtype2rtype['left'][left_msubtype] = {rtype}
            # eg, 'entity:ORG' -> ('relation:--ORG-AFF-->', ...) from backward
            if right_msubtype in msubtype2rtype['right']:
                msubtype2rtype['right'][right_msubtype] |= {rtype}
            else:
                msubtype2rtype['right'][right_msubtype] = {rtype}

    # add in ALL for misclassification at the beginning
    mtype2msubtype['<UNK>'] = mtype2msubtype.values()[0]
    return mtype2msubtype, msubtype2rtype

def compute_mentions(doc, fine_grained=False, omit_timex=True):
    """ Compute gold mentions as (*span, type) from yaat.

    Return them sorted lexicographicaly. """
    mentions = []
    for ann in doc['annotations']:
        if ann['ann-type'] == u'node':
            if omit_timex and ann['node-type'] == 'value':
                continue
            mention_type = ann['node-type']+':'+ann['type']
            # always use event anchor subtypes (needed for correct roles)
            if fine_grained or ann['node-type'] == 'event-anchor':
                mention_type += ':'+ann['subtype']
            mentions.append((ann['ann-span'][0], ann['ann-span'][1], mention_type))
    return sorted(mentions, key=lambda x: x[:2])

def compute_relations(doc, mentions):
    """ Compute gold relations as (*left_span, *right_span, type) from yaat """
    relations = []
    mention_spans = { m[:2] for m in mentions }
    id2ann = { ann['ann-uid']:ann for ann in doc['annotations']}
    for ann in doc['annotations']:
        if ann['ann-type'] == u'edge':
            left_span = tuple(id2ann[ann['ann-left']]['ann-span'])
            right_span = tuple(id2ann[ann['ann-right']]['ann-span'])
            # omit relations that we didn't include spans for
            if left_span not in mention_spans or right_span not in mention_spans:
                continue
            if 'ARG' in ann['type']: # event args go by subtypes
                rel_type = ann['edge-type']+':'+ann['type']
            else:
                rel_type = ann['edge-type']+':'+ann['type']
            relations.append((left_span[0], left_span[1], right_span[0], right_span[1], rel_type))
    return relations

def compute_max_dists(docs):
    max_rel_dist = 0
    for doc in docs:
        for ann in doc['annotations']:
            if ann['ann-type'] == 'edge' and int(ann['dist']) > max_rel_dist:
                max_rel_dist = int(ann['dist'])
    return max_rel_dist

def resolve_mentions_and_relations(annotations):
    """ Remove any duplicate, nested, or overlapping mentions and their relations."""
    def overlaps(node, qnode):
        nspan = node['ann-span']
        qspan = qnode['ann-span']
        if nspan[0] == qspan[0] or nspan[1] == qspan[1]: # sharing a boundary => overlap
            return True
        elif nspan[0] > qspan[0] and nspan[0] < qspan[1]: # [  (  ]
            return True
        elif (nspan[1]-1) > qspan[0] and nspan[1] < qspan[1]: # [  )  ]
            return True
        else:
            return False

    # sort all of the mentions by width
    # and resolve them by preference to width
    # so if we find a mention that is within or overlaps
    # a mention we've previously seen, omit it
    sorted_nodes = sorted([ a for a in annotations
                           if a['ann-type'] == u'node'],
                           key=lambda x: x['ann-span'][1] - x['ann-span'][0],
                           reverse=True)
    resolved_annotations = []
    for node in sorted_nodes:
        if not any([ overlaps(node, qnode) for qnode in resolved_annotations]):
            resolved_annotations.append(node)

    # now only include relations whose constiuent mentions are still around
    node_id_set = set([node['ann-uid'] for node in resolved_annotations ])
    for ann in annotations:
        if ann['ann-type'] == u'edge':
            if ann['ann-left'] in node_id_set and ann['ann-right'] in node_id_set:
                resolved_annotations.append(ann)
    return resolved_annotations

def get_ace_extraction_data(count=0,
                            splits_dir='data/ACE 2005/splits/',
                            data_dir='data/ACE 2005/yaat/',
                            map_func_name='NoVal_BIO_map',
                            train_vocab_only=False,
                            oversample_unks=True,
                            v=1):
    if v > 0:
        print "Loading data..."
    if not splits_dir.endswith('/'): splits_dir += '/'
    if not data_dir.endswith('/'): data_dir += '/'
    train_set = open(splits_dir+'train.txt', 'r').read().split()
    dev_set = open(splits_dir+'dev.txt', 'r').read().split()
    test_set = open(splits_dir+'test.txt', 'r').read().split()

    train_docs = [ json.loads(io.open(data_dir+f+'.yaat', 'r').read()) for f in train_set ]
    dev_docs = [ json.loads(io.open(data_dir+f+'.yaat', 'r').read()) for f in dev_set]
    test_docs = [ json.loads(io.open(data_dir+f+'.yaat', 'r').read()) for f in test_set]
    for doc, fname in zip(train_docs+dev_docs+test_docs, train_set+dev_set+test_set):
        doc.update({'fname':fname})
    # train_data = json.loads(io.open('data/ace_05_head_yaat_train.json', 'r').read())
    # dev_data = json.loads(io.open('data/ace_05_head_yaat_dev.json', 'r').read())
    # test_data = json.loads(io.open('data/ace_05_head_yaat_test.json', 'r').read())
    # all_data_values = train_data.values() + dev_data.values() + test_data.values()

    # get vocabs and generate info network annotations
    token_vocab = Vocab(min_count=count)
    pos_vocab = Vocab(min_count=0)
    boundary_vocab = Vocab(min_count=0)
    mention_vocab = Vocab(min_count=0)
    relation_vocab = Vocab(min_count=0)
    for dataset_i, docs in enumerate([train_docs, dev_docs, test_docs]):
        for doc in docs:
            if train_vocab_only:
                if dataset_i ==0:
                    token_vocab.add(doc['tokens'])
            else:
                token_vocab.add(doc['tokens'])
            doc['pos'] = [ ann['type'] for ann in doc['annotations']
                           if ann['ann-type'] == 'node'
                           and ann['node-type'] == 'pos' ]
            pos_vocab.add(doc['pos'])

            # get rid of pos from annotations before dealing with nestings
            doc['annotations'] = [ ann for ann in doc['annotations']
                                   if (ann['ann-type'] == 'node'
                                       and ann['node-type'] != 'pos')
                                      or ann['ann-type'] == 'edge' ]

            # dedupe and remove nestings/overlaps
            doc['annotations'] = resolve_mentions_and_relations(doc['annotations'])

            # boundary labels
            map_func = globals()[map_func_name]
            doc['boundary_labels'] = compute_flat_mention_labels(doc, map_func)
            if dataset_i==0: # use only train data to estimate vocabs
                boundary_vocab.add(doc['boundary_labels'])

            # mention labels
            doc['mentions'] = compute_mentions(doc)
            if dataset_i==0: # use only train data to estimate vocabs
                mention_vocab.add([m[2] for m in doc['mentions']])

            # relation labels
            relations = compute_relations(doc, doc['mentions'])
            # print '{} mentions, {} relations'.format(len(doc['mentions']), len(relations))

            # add in NULL relations for all mention pairs that don't have a relation
            rel_mentions = set([ r[:4] for r in relations])
            seen_mentions = set()
            n_null, n_real = 0, 0
            for i, m1 in enumerate(doc['mentions']):
                for m2 in doc['mentions'][i+1:]:
                    r = m1[:2] + m2[:2]
                    if r not in rel_mentions:
                        relations.append(r+(u'--NULL--',))
                        n_null += 1
                    else:
                        n_real += 1
                        seen_mentions |= set([r])
            assert n_real+n_null == int(comb(len(doc['mentions']), 2)),\
                    "There should always be m choose 2 relations"
            doc['relations'] = relations
            if dataset_i==0: # use only train data to estimate vocabs
                relation_vocab.add([r[4] for r in doc['relations']])

        # estimate type constraints using only train data
        if dataset_i == 0:
            mtype2msubtype, msubtype2rtype = compute_typemaps(docs)
    # compute the typing stats for extract all mentions
    tag_map = compute_tag_map(boundary_vocab)

    # create datasets
    x_train = [ doc['tokens'] for doc in train_docs ]
    x_dev = [ doc['tokens'] for doc in dev_docs ]
    x_test = [ doc['tokens'] for doc in test_docs ]
    p_train = [ doc['pos'] for doc in train_docs ]
    p_dev = [ doc['pos'] for doc in dev_docs ]
    p_test = [ doc['pos'] for doc in test_docs ]
    b_train = [ doc['boundary_labels'] for doc in train_docs ]
    b_dev = [ doc['boundary_labels'] for doc in dev_docs ]
    b_test = [ doc['boundary_labels'] for doc in test_docs ]
    m_train = [ doc['mentions'] for doc in train_docs ]
    m_dev = [ doc['mentions'] for doc in dev_docs ]
    m_test = [ doc['mentions'] for doc in test_docs ]
    r_train = [ doc['relations'] for doc in train_docs ]
    r_dev = [ doc['relations'] for doc in dev_docs ]
    r_test = [ doc['relations'] for doc in test_docs ]
    # keep track of doc file names
    f_train = [ doc['fname'] for doc in train_docs ]
    f_dev = [ doc['fname'] for doc in dev_docs ]
    f_test = [ doc['fname'] for doc in test_docs ]

    # before converting drop infrequents
    token_vocab.drop_infrequent()
    pos_vocab.drop_infrequent()
    boundary_vocab.drop_infrequent()
    mention_vocab.drop_infrequent()
    relation_vocab.drop_infrequent()

    if v > 1:
        print "POS vocab:"
        for i, v in pos_vocab._idx2vocab.items():
            print '\t{}:: {}'.format(i,v)

        print "Boundary vocab:"
        for i, v in boundary_vocab._idx2vocab.items():
            print '\t{}:: {}'.format(i,v)

        print "Mention vocab:"
        for i, v in mention_vocab._idx2vocab.items():
            print '\t{}:: {}'.format(i,v)
        #
        print "Relation vocab:"
        for i, v in relation_vocab._idx2vocab.items():
            print '\t{}:: {}'.format(i,v)
    if v > 2:
        print "Tags to Mention types:"
        for t, m in tag_map['tag2mtype'].items():
            print '\t{}:: {}'.format(t,m)

        print "Mention types to subtypes:"
        for m, s in mtype2msubtype.items():
            print '\t{}:'.format(m)
            for t in s:
                print '\t  {}'.format(t)

        print "Mention subtypes to relations:"
        for m, s in msubtype2rtype.items():
            print '\t{}:'.format(m)
            for t, v in s.items():
                print '\t  {}:: {}'.format(t, v)

    # convert to indices in vocab
    ix_train = convert_sequences(x_train, token_vocab.idx)
    ix_dev = convert_sequences(x_dev, token_vocab.idx)
    ix_test = convert_sequences(x_test, token_vocab.idx)
    ip_train = convert_sequences(p_train, pos_vocab.idx)
    ip_dev = convert_sequences(p_dev, pos_vocab.idx)
    ip_test = convert_sequences(p_test, pos_vocab.idx)
    ib_train = convert_sequences(b_train, boundary_vocab.idx)
    ib_dev = convert_sequences(b_dev, boundary_vocab.idx)
    ib_test = convert_sequences(b_test, boundary_vocab.idx)
    convert_mention = lambda x: x[:-1]+(mention_vocab.idx(x[-1]),) # type is last
    im_train = convert_sequences(m_train, convert_mention)
    im_dev = convert_sequences(m_dev, convert_mention)
    im_test = convert_sequences(m_test, convert_mention)
    convert_relation = lambda x: x[:-1]+(relation_vocab.idx(x[-1]),) # type is last
    ir_train = convert_sequences(r_train, convert_relation)
    ir_dev = convert_sequences(r_dev, convert_relation)
    ir_test = convert_sequences(r_test, convert_relation)
    if v > 0:
        print '{} train, {} dev, and {} test documents'.format(len(x_train), len(x_dev), len(x_test))

    # oversample unks in the train data so it has the same ratio as the dev data
    if oversample_unks:
        x_t = [token_vocab.token(i) for xs in ix_train for i in xs]
        x_d = [token_vocab.token(i) for xs in ix_dev for i in xs]
        train_unks_ratio = float(len([x for x in x_t if x == '<UNK>']))/len(x_t)
        dev_unks_ratio = float(len([x for x in x_d if x == '<UNK>']))/len(x_d)
        if v > 0:
            print "Oversampling training unks from {}% to {}%".format(
                train_unks_ratio*100, dev_unks_ratio*100)
        for i in range(len(ix_train)):
            for j in range(len(ix_train[i])):
                if random.uniform(0,1) < dev_unks_ratio-train_unks_ratio:
                    ix_train[i][j] = token_vocab.iunk

        x_t = [token_vocab.token(i) for xs in ix_train for i in xs]
        x_d = [token_vocab.token(i) for xs in ix_dev for i in xs]
        train_unks_ratio = float(len([x for x in x_t if x == '<UNK>']))/len(x_t)
        dev_unks_ratio = float(len([x for x in x_d if x == '<UNK>']))/len(x_d)
        if v > 0:
            print "Resulting sampled ratios: Train:{}%, Dev:{}%".format(
                train_unks_ratio*100, dev_unks_ratio*100)

    dataset = { # vocabs
                'token_vocab':token_vocab,
                'pos_vocab':pos_vocab,
                'boundary_vocab':boundary_vocab,
                'mention_vocab':mention_vocab,
                'relation_vocab':relation_vocab,
                # label compaitbilities for different levels of ie tasks
                'tag_map':tag_map,
                'mtype2msubtype':mtype2msubtype,
                'msubtype2rtype':msubtype2rtype,
                # tokenized data and labels
                'x_train':x_train,
                'p_train':p_train,
                'b_train':b_train,
                'm_train':m_train,
                'r_train':r_train,
                'x_dev':x_dev,
                'p_dev':p_dev,
                'b_dev':b_dev,
                'm_dev':m_dev,
                'r_dev':r_dev,
                'x_test':x_test,
                'p_test':p_test,
                'b_test':b_test,
                'm_test':m_test,
                'r_test':r_test,
                # data converted to indices
                'ix_train':ix_train,
                'ip_train':ip_train,
                'ib_train':ib_train,
                'im_train':im_train,
                'ir_train':ir_train,
                'ix_dev':ix_dev,
                'ip_dev':ip_dev,
                'ib_dev':ib_dev,
                'im_dev':im_dev,
                'ir_dev':ir_dev,
                'ix_test':ix_test,
                'ip_test':ip_test,
                'ib_test':ib_test,
                'im_test':im_test,
                'ir_test':ir_test,
                # names of input docs and docs themselves
                'f_train':f_train,
                'f_dev':f_dev,
                'f_test':f_test,
                # dists
                'max_r_dist':compute_max_dists(train_docs)
            }

    print 'train {}, dev {}, test {} max dists'.format(
        compute_max_dists(train_docs), compute_max_dists(dev_docs), compute_max_dists(test_docs))
    return dataset
