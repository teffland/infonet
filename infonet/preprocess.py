import json
import io
from scipy.misc import comb # to check for num relations

from infonet.vocab import Vocab
from infonet.util import convert_sequences

# NOTE: all mapping schemes assume the annotation spans correspond with python slice indexing
# eg ann-span = [0,2] means the interval [0,2) or tokens[0:2]

def Entity_BIO_map(mention_labels, annotation):
    """ Uses BIO scheme (untyped) for entities only """
    if annotation['node-type'] == 'entity':
        left, right = tuple(annotation['ann-span'])
        mention_labels[left] = 'B'
        for i in range(1, right-left):
            mention_labels[left+i] = 'I'
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
        mention_labels[left] = 'B-'+mention_type
        for i in range(1, right-left):
            mention_labels[left+i] = 'I-'+mention_type
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
    """ Uses BIO scheme (typed) for entities only """
    if annotation['node-type'] in ('entity', 'event-anchor'):
        mention_type = annotation['type']
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
            mention_labels[left] = 'U-'+mention_type
        else:
            mention_labels[left] = 'B-'+mention_type
            for i in range(1, right-left-1):
                mention_labels[left+i] = 'I-'+mention_type
            mention_labels[right-1] = 'L-'+mention_type
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
                    else 'ALL'
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

def resolve_annotations(annotations):
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
                           if a['ann-type'] ==u'node'],
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

def get_ace_extraction_data(count=0, map_func_name='NoVal_BIO_map', **kwds):
    print "Loading data..."
    # load data
    train_data = json.loads(io.open('data/ace_05_head_yaat_train.json', 'r').read())
    dev_data = json.loads(io.open('data/ace_05_head_yaat_dev.json', 'r').read())
    test_data = json.loads(io.open('data/ace_05_head_yaat_test.json', 'r').read())
    # all_data_values = train_data.values() + dev_data.values() + test_data.values()

    # get vocabs and generate info network annotations
    token_vocab = Vocab(min_count=count)
    boundary_vocab = Vocab(min_count=0)
    mention_vocab = Vocab(min_count=0)
    relation_vocab = Vocab(min_count=0)
    for dataset_i, docs in enumerate([train_data.values(), dev_data.values(), test_data.values()]):
        for doc in docs:
            if dataset_i==0: # use only train data to estimate vocabs
                token_vocab.add(doc['tokens'])

            # dedupe and remove nestings/overlaps
            doc['annotations'] = resolve_annotations(doc['annotations'])

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
    x_train = [ doc['tokens'] for doc in train_data.values() ]
    x_dev = [ doc['tokens'] for doc in dev_data.values() ]
    x_test = [ doc['tokens'] for doc in test_data.values() ]
    b_train = [ doc['boundary_labels'] for doc in train_data.values() ]
    b_dev = [ doc['boundary_labels'] for doc in dev_data.values() ]
    b_test = [ doc['boundary_labels'] for doc in test_data.values() ]
    m_train = [ doc['mentions'] for doc in train_data.values() ]
    m_dev = [ doc['mentions'] for doc in dev_data.values() ]
    m_test = [ doc['mentions'] for doc in test_data.values() ]
    r_train = [ doc['relations'] for doc in train_data.values() ]
    r_dev = [ doc['relations'] for doc in dev_data.values() ]
    r_test = [ doc['relations'] for doc in test_data.values() ]

    # before converting drop infrequents
    token_vocab.drop_infrequent()
    boundary_vocab.drop_infrequent()
    mention_vocab.drop_infrequent()
    relation_vocab.drop_infrequent()

    # print "Boundary vocab:"
    # for i, v in boundary_vocab._idx2vocab.items():
    #     print '\t{}:: {}'.format(i,v)
    #
    # print "Mention vocab:"
    # for i, v in mention_vocab._idx2vocab.items():
    #     print '\t{}:: {}'.format(i,v)
    #
    # print "Relation vocab:"
    # for i, v in relation_vocab._idx2vocab.items():
    #     print '\t{}:: {}'.format(i,v)
    #
    # print "Tags to Mention types:"
    # for t, m in tag_map['tag2mtype'].items():
    #     print '\t{}:: {}'.format(t,m)
    #
    # print "Mention types to subtypes:"
    # for m, s in mtype2msubtype.items():
    #     print '\t{}:'.format(m)
    #     for t in s:
    #         print '\t  {}'.format(t)
    #
    # print "Mention subtypes to relations:"
    # for m, s in msubtype2rtype.items():
    #     print '\t{}:'.format(m)
    #     for t, v in s.items():
    #         print '\t  {}:: {}'.format(t, v)

    # convert to indices in vocab
    ix_train = convert_sequences(x_train, token_vocab.idx)
    ix_dev = convert_sequences(x_dev, token_vocab.idx)
    ix_test = convert_sequences(x_test, token_vocab.idx)
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
    print '{} train, {} dev, and {} test documents'.format(len(x_train), len(x_dev), len(x_test))

    dataset = { # vocabs
                'token_vocab':token_vocab,
                'boundary_vocab':boundary_vocab,
                'mention_vocab':mention_vocab,
                'relation_vocab':relation_vocab,
                # label compaitbilities for different levels of ie tasks
                'tag_map':tag_map,
                'mtype2msubtype':mtype2msubtype,
                'msubtype2rtype':msubtype2rtype,
                # tokenized data
                'x_train':x_train,
                'b_train':b_train,
                'm_train':m_train,
                'r_train':r_train,
                'x_dev':x_dev,
                'b_dev':b_dev,
                'm_dev':m_dev,
                'r_dev':r_dev,
                'x_test':x_test,
                'b_test':b_test,
                'm_test':m_test,
                'r_test':r_test,
                # data converted to indices
                'ix_train':ix_train,
                'ib_train':ib_train,
                'im_train':im_train,
                'ir_train':ir_train,
                'ix_dev':ix_dev,
                'ib_dev':ib_dev,
                'im_dev':im_dev,
                'ir_dev':ir_dev,
                'ix_test':ix_test,
                'ib_test':ib_test,
                'im_test':im_test,
                'ir_test':ir_test
            }
    return dataset
