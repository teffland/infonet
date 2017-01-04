""" ETL to extract the ACE 2005 English corpus to yaat json format """
from glob import glob
from io import open
from time import time
from bs4 import BeautifulSoup as bs
import spacy
import json


######################
### EXTRACTOR DEFS ###
######################
def extract_anchor2offset(annotation_soup):
    """ Each span in ACE is indexed in the doc by anchor, which then has the offset.

    This creates a map from anchors to offsets.
    """
    anchor2offset = {}
    for a in annotation_soup.find_all('anchor'):
        if 'offset' in a.attrs:
            offset = int(float(a['offset']))
            anchor2offset[a['id']] = offset
        else:
            anchor2offset[a['id']] = None
    return anchor2offset

def mention2str(node, tokens):
    # check if the tokens are actually just all characters
    if all([len(token)==1 for token in tokens]):
        joinstr = ""
    else:
        joinstr = " "
    return '"{}":{}:{}'.format(joinstr.join(tokens[node['ann-span'][0]:node['ann-span'][1]]),
                              node['node-type'], node['type'])

def relation2str(edge, node_map, tokens):
    e1 = mention2str(node_map[edge['ann-left']], tokens)
    e2 = mention2str(node_map[edge['ann-right']], tokens)
    return '({}){}({}) (d={})'.format(e1, edge['type'], e2, edge['dist'])

def extract_value_nodes(annotations, anchor2offset):
    """ Extract the value nodes from the ACE document """
    values = [ a for a in annotations if a['type']=='value']
    value_mentions = [ a for a in annotations if a['type']=='value_mention']
    value_nodes = []
    for value in values:
        for value_mention in value_mentions:
            parent_value_id = value_mention.find('feature', {'name':'value'}).text
            if parent_value_id == value['id']:
                value_span = (anchor2offset[value_mention['start']],
                              anchor2offset[value_mention['end']])
                node = {u'ann-type':u'node',
                        u'ann-uid':value_mention['id'],
                        u'ann-span':value_span,
                        u'node-type':u'value'}

                for feature in value.find_all('feature'):
                    node[feature['name']] = feature.text
                value_nodes.append(node)
    return value_nodes

def extract_entity_nodes(annotations, anchor2offset, use_head_span=True, tokens=None):
    """ Extract the entity nodes from ACE

    If use_head, create entity nodes using their heads for spans.
      This follows the official ACE evaluation criterion for entity boundaries.
    """
    entities = [ a for a in annotations if a['type'] == 'entity']
    entity_mentions = [ a for a in annotations if a['type'] == 'entity_mention']
    if use_head_span:
        head_mentions = [ a for a in annotations if a['type'] == 'entity_mention_head']

    entity_nodes = []
    for entity in entities:
        if tokens: print '-'*50
        for entity_mention in entity_mentions:
            parent_entity_id = entity_mention.find('feature', {'name':'entity'}).text
            if parent_entity_id == entity['id']:
                if use_head_span:
                    head = entity_mention.find('feature', {'name':'head'})
                    if head is not None:
                        head_id = head.text
                        mention_span = None
                        for head_mention in head_mentions:
                            if head_mention['id'] == head_id:
                                mention_span = (anchor2offset[head_mention['start']],
                                                anchor2offset[head_mention['end']])
                        if mention_span is None:
                            print "EMPTY HEAD FOR MENTION: {}, USING FULL SPAN".format(entity_mention)
                            mention_span = (anchor2offset[entity_mention['start']],
                                            anchor2offset[entity_mention['end']])
                    else:
                        print "NO HEAD FOR MENTION: {}, USING FULL SPAN".format(entity_mention)
                        mention_span = (anchor2offset[entity_mention['start']],
                                        anchor2offset[entity_mention['end']])

                else:
                    mention_span = (anchor2offset[entity_mention['start']],
                                    anchor2offset[entity_mention['end']])
                node = {u'ann-type':u'node',
                        u'ann-uid':entity_mention['id'],
                        u'ann-span':mention_span,
                        u'node-type':u'entity'}
                # pull out entity features
                for feature in entity.find_all('feature'):
                    node[feature['name']] = feature.text
                # pull out mention features
                for feature in entity_mention.find_all('feature'):
                    if feature['name'] not in set(['head', 'indexing_type']):
                        node[feature['name'].replace('type', 'mention_type')] = feature.text
                if tokens: print '*', mention2str(node, tokens)
                entity_nodes.append(node)
    return entity_nodes

def extract_event_anchor_nodes(annotations, anchor2offset, tokens=None):
    """ Extract the event anchor mentions """
    events = [ a for a in annotations if a['type'] == 'event']
    event_mentions = [ a for a in annotations if a['type'] == 'event_mention']
    event_mention_anchors = [ a for a in annotations if a['type'] == 'event_mention_anchor']
    event_anchor_nodes = []
    for event in events:
        if tokens: print '='*80
        for event_mention in event_mentions:
            parent_event_id = event_mention.find('feature', {'name':'event'}).text
            if parent_event_id == event['id']:
                if tokens: print '-'*80
                child_anchor_id = event_mention.find('feature', {'name':'anchor'}).text
                for event_mention_anchor in event_mention_anchors:
                    if child_anchor_id == event_mention_anchor['id']:
                        anchor_span = (anchor2offset[event_mention_anchor['start']],
                                       anchor2offset[event_mention_anchor['end']])
                        node = {u'ann-type':u'node',
                                u'ann-uid':event_mention_anchor['id'],
                                u'ann-span':anchor_span,
                                u'node-type':u'event-anchor',
                                u'event-mention':event_mention['id']
                                }
                        # pull out the event features
                        for feature in event.find_all('feature'):
                            node[feature['name']] = feature.text

                        # pull out the event_mention features
                        for feature in event_mention.find_all('feature'):
                            if feature['name'] not in set(['anchor', 'indexing_type']):
                                node[feature['name']] = feature.text

                        if tokens: print mention2str(node, tokens)
                        event_anchor_nodes.append(node)
    return event_anchor_nodes

def edge_dist(edge, node_map):
    """ Calculate the distance between the two arguments of an edge
    as the difference between the left boundaries of the argument spans
    """
    return (node_map[edge['ann-right']]['ann-span'][0]
            - node_map[edge['ann-left']]['ann-span'][0])

def extract_relation_edges(annotations, node_map, tokens=None):
    """ Extraction relation egdes """
    relations = [ a for a in annotations if a['type'] == 'relation']
    relation_mentions = [ a for a in annotations if a['type'] == 'relation_mention']
    relation_mention_arguments = [ a for a in annotations if a['type'] == 'relation_mention_argument']
    edges = []
    for relation in relations:
        for relation_mention in relation_mentions:
            parent_relation_id = relation_mention.find('feature', {'name':'relation'}).text
            if parent_relation_id == relation['id']:
                edge = {u'ann-type':u'edge',
                        u'ann-uid':relation_mention['id'],
                        u'edge-type':u'relation'}

                # pull out the relation features
                for feature in relation.find_all('feature'):
                    edge[feature['name']] = feature.text

                # find the two arguments
                nominals = [None, None]
                for relation_mention_arg in relation_mention_arguments:
                    parent_relation_mention_id = relation_mention_arg.find('feature', {'name':'relation_mention'}).text
                    if parent_relation_mention_id == relation_mention['id']:
                        argnum = int(relation_mention_arg.find('feature', {'name':'argnum'}).text)-1
                        node_id = relation_mention_arg.find('feature', {'name':'object_mention'}).text
#                         print argnum,
#                         print node_map[node_id]['type']
                        # some relations have a 3rd argument that is a time value
                        if argnum == 2:
                            edge[u'time-arg'] = node_id
                            continue
                        nominals[argnum] = node_id

                # now decide the left and right (lexically, not by relation direction)
                # by ordering the left end of their spans
                # breaking ties by the lexical order of the right end
                if node_map[nominals[0]]['ann-span'][0] < node_map[nominals[1]]['ann-span'][0]:
                    # arg1.left comes before arg2.left
                    edge['ann-left'] = nominals[0]
                    edge['ann-right'] = nominals[1]
                    edge['type'] = u'--{}->'.format(edge['type'])
                    edge['subtype'] = u'--{}->'.format(edge['subtype'])
                elif node_map[nominals[1]]['ann-span'][0] < node_map[nominals[0]]['ann-span'][0]:
                    # arg2.left comes before arg1.left
                    edge['ann-left'] = nominals[1]
                    edge['ann-right'] = nominals[0]
                    edge['type'] = u'<-{}--'.format(edge['type'])
                    edge['subtype'] = u'<-{}--'.format(edge['subtype'])
                elif node_map[nominals[0]]['ann-span'][1] < node_map[nominals[1]]['ann-span'][1]:
                    # arg1.right comes before arg2.right
                    edge['ann-left'] = nominals[0]
                    edge['ann-right'] = nominals[1]
                    edge['type'] = u'--{}->'.format(edge['type'])
                    edge['subtype'] = u'--{}->'.format(edge['subtype'])
                elif node_map[nominals[1]]['ann-span'][1] < node_map[nominals[0]]['ann-span'][1]:
                    # arg2.right comes before arg1.right
                    edge['ann-left'] = nominals[1]
                    edge['ann-right'] = nominals[0]
                    edge['type'] = u'<-{}--'.format(edge['type'])
                    edge['subtype'] = u'<-{}--'.format(edge['subtype'])
                else:
                    # arg1 and arg2 occupy identical spans. just use the order given
                    print "RELATION SAME SPANS E1:{}, E2:{}".format(nominals[0], nominals[1])
                    edge['ann-left'] = nominals[0]
                    edge['ann-right'] = nominals[1]
                    edge['type'] = u'--{}->'.format(edge['type'])
                    edge['subtype'] = u'--{}->'.format(edge['subtype'])
                edge['dist'] = edge_dist(edge, node_map)
                edges.append(edge)
                if tokens: print relation2str(edge, node_map, tokens)
    return edges

def extract_event_participant_edges(annotations, node_map, tokens=None):
    """ Extract all of the edges representing event arguments """
    event_mention_participants = [ a for a in annotations if a['type'] == 'event_mention_participant']
    event_mention2event_anchor = {node['event-mention']:node['ann-uid'] for node in node_map.values()
                                  if node['node-type'] == 'event-anchor'}
    edges = []
    for participant in event_mention_participants:
        event_mention_id = participant.find('feature', {'name':'event_mention'}).text
        event_mention_anchor_id = event_mention2event_anchor[event_mention_id]
        entity_mention_id = participant.find('feature', {'name':'object_mention'}).text
        edge = {u'ann-type':u'edge',
                u'ann-uid':participant['id'],
                u'edge-type':u'event-argument'}
        event_node = node_map[event_mention_anchor_id]
        participant_node = node_map[entity_mention_id]
        role = participant.find('feature', {'name':'role'}).text
        subrole = participant.find('feature', {'name':'subrole'}).text

        # now decide the left and right (lexically, not by relation direction)
        # by ordering the left end of their spans
        # breaking ties by the lexical order of the right end
        if node_map[event_mention_anchor_id]['ann-span'][0] < node_map[entity_mention_id]['ann-span'][0]:
            # event_anchor.left comes before participant.left
            edge['ann-left'] = event_mention_anchor_id
            edge['ann-right'] = entity_mention_id
            edge['type'] = u'--ARG:{}->'.format(role)
            edge['subtype'] = u'--ARG:{}->'.format(subrole)
        elif node_map[entity_mention_id]['ann-span'][0] < node_map[event_mention_anchor_id]['ann-span'][0]:
            # participant.left comes before event_anchor.left
            edge['ann-left'] = entity_mention_id
            edge['ann-right'] = event_mention_anchor_id
            edge['type'] = u'<-ARG:{}--'.format(role)
            edge['subtype'] = u'<-ARG:{}--'.format(subrole)
        elif node_map[event_mention_anchor_id]['ann-span'][1] < node_map[entity_mention_id]['ann-span'][1]:
            # event_anchor.right comes before participant.right
            edge['ann-left'] = event_mention_anchor_id
            edge['ann-right'] = entity_mention_id
            edge['type'] = u'--ARG:{}->'.format(role)
            edge['subtype'] = u'--ARG:{}->'.format(subrole)
        elif node_map[entity_mention_id]['ann-span'][1] < node_map[event_mention_anchor_id]['ann-span'][1]:
            # participant.right comes before event_anchor.right
            edge['ann-left'] = entity_mention_id
            edge['ann-right'] = event_mention_anchor_id
            edge['type'] = u'<-ARG:{}--'.format(role)
            edge['subtype'] = u'<-ARG:{}--'.format(subrole)
        else:
            # arg1 and arg2 occupy identical spans. just use the order given
            print "EVENT PARTICIPANT SAME SPANS EV:{}, ARG:{}".format(event_mention_anchor_id, entity_mention_id)
            edge['ann-left'] = event_mention_anchor_id
            edge['ann-right'] = entity_mention_id
            edge['type'] = u'--ARG:{}->'.format(role)
            edge['subtype'] = u'--ARG:{}->'.format(subrole)
        edge['dist'] = edge_dist(edge, node_map)
        edges.append(edge)
        if tokens: print relation2str(edge, node_map, tokens)
    return edges

def extract_coreferent_edges(annotations, node_map, tokens=None):
    """ Synthetically create an edge for each pair of coreferent mentions """
    entities = [ a for a in annotations if a['type'] == 'entity']
    entity_mentions = [ a for a in annotations if a['type'] == 'entity_mention']

    edges = []
    for i, mention1 in enumerate(entity_mentions):
        for mention2 in entity_mentions[i+1:]:
            e1_id = mention1.find('feature', {'name':'entity'}).text
            e2_id = mention2.find('feature', {'name':'entity'}).text
            if e1_id == e2_id:
                em1_id = mention1['id']
                em2_id = mention2['id']
                edge = {u'ann-type':u'edge',
                        u'ann-uid':'{}={}'.format(e1_id, e2_id),
                        u'edge-type':u'coreference',
                        u'type':u'--SameAs--'}
                # now decide the left and right (lexically, not by relation direction)
                # by ordering the left end of their spans
                # breaking ties by the lexical order of the right end
                if node_map[em1_id]['ann-span'][0] < node_map[em2_id]['ann-span'][0]:
                    # em1.left comes before em2.left
                    edge['ann-left'] = em1_id
                    edge['ann-right'] = em2_id
                elif node_map[em2_id]['ann-span'][0] < node_map[em1_id]['ann-span'][0]:
                    # em2.left comes before em1.left
                    edge['ann-left'] = em2_id
                    edge['ann-right'] = em1_id
                elif node_map[em1_id]['ann-span'][1] < node_map[em2_id]['ann-span'][1]:
                    # em1.right comes before em2.right
                    edge['ann-left'] = em1_id
                    edge['ann-right'] = em2_id
                elif node_map[em2_id]['ann-span'][1] < node_map[em1_id]['ann-span'][1]:
                    # em2.right comes before em1.right
                    edge['ann-left'] = em2_id
                    edge['ann-right'] = em1_id
                else:
                    # two identical spans. don't do coref here
#                     continue
                    print "COREFERENT SAME SPANS, EM1:{}, EM2:{}".format(em1_id, em2_id)
                    edge['ann-left'] = em1_id
                    edge['ann-right'] = em2_id
                edge['dist'] = edge_dist(edge, node_map)
                edges.append(edge)
                if tokens: print relation2str(edge, node_map, tokens)
    return edges

def convert_charspans_to_tokenspans(nodes, spacy_doc):
    """ create a function that buckets char idxs to token idxs"""
    char2token_idxmap = {}
    tokens = [] # the tokenization of the document as list(unicode)
    j = 0
    for token_idx, token in enumerate(spacy_doc):
        token_width = len(token) + len(token.whitespace_)
        tokens.append(token.text)
        for i in range(token_width):
            char2token_idxmap[j] = token_idx
            j +=1

    def charspan2tokenspan((char_start, char_end)):
        # add 1 to support python style indexing
        return (char2token_idxmap[char_start], char2token_idxmap[char_end]+1)

    # now convert the charspans in the nodes
    for node in nodes:
        node['ann-span'] = charspan2tokenspan(node['ann-span'])

    return nodes, tokens

def extract_doc(text_f, ann_f, spacy_pipe, use_head_span=True):
    """ Do all of the extractions for a document """
    # read in the source and annotation docs
    text = open(text_f, 'r').read()
    annotation_soup = bs(open(ann_f, 'r').read(), "html.parser")
    doc_id = annotation_soup.find('metadataelement',{'name':'docid'}).text


    # create a map of anchors to character offsets
    anchor2offset = extract_anchor2offset(annotation_soup)

    # extract the annotation graph
    annotations = annotation_soup.find_all('annotation')
    nodes = []
    nodes.extend(extract_value_nodes(annotations, anchor2offset))
    nodes.extend(extract_entity_nodes(annotations, anchor2offset,
                                      use_head_span=use_head_span))
    nodes.extend(extract_event_anchor_nodes(annotations, anchor2offset))
    node_map = {node['ann-uid']:node for node in nodes}
    edges = []
    edges.extend(extract_relation_edges(annotations, node_map))
    edges.extend(extract_event_participant_edges(annotations, node_map))
    edges.extend(extract_coreferent_edges(annotations, node_map))

    # replace newlines with spaces so spacy doesn't count them as tokens
    # but the char offsets are still correct
    text = text.replace('\n', ' ')
    # now tokenize the document and convert annotation charspans to token spans
    doc = spacy_pipe(text)
    nodes, tokens = convert_charspans_to_tokenspans(nodes, doc)
    return doc_id, text, tokens, nodes, edges

if __name__ == '__main__':
    print "Loading Spacy..."
    nlp = spacy.load('en')

    text_files = glob('data/ACE 2005/data/English/*/timex2norm/*sgm')
    annotation_files = glob('data/ACE 2005/data/English/*/timex2norm/*ag.xml')

    # load in the fnames from the Li/Miwa's splits
    train_set = set(open('data/ACE 2005/splits/train.txt', 'r').read().split())
    dev_set = set(open('data/ACE 2005/splits/dev.txt', 'r').read().split())
    test_set = set(open('data/ACE 2005/splits/test.txt', 'r').read().split())

    n = len(text_files)

    train_annotations = {}
    dev_annotations = {}
    test_annotations = {}

    start = time()
    for doc_i, (text_f, ann_f) in enumerate(zip(text_files, annotation_files)):
        print '\rExtracting Document {} / {} : {}'.format(doc_i, n, "/".join(ann_f.split('/')[-3:]))
        doc_id, text, tokens, nodes, edges = extract_doc(text_f, ann_f, nlp)
        fname = ann_f.split('/')[-1][:-7]
        # print ann_f, fname
        if fname in train_set:
            train_annotations[doc_id] = {'text':text,
                                       'tokens':tokens,
                                       'annotations':nodes+edges}
        elif fname in dev_set:
            dev_annotations[doc_id] = {'text':text,
                                       'tokens':tokens,
                                       'annotations':nodes+edges}
        elif fname in test_set:
            test_annotations[doc_id] = {'text':text,
                                       'tokens':tokens,
                                       'annotations':nodes+edges}
        else:
            continue

    total = time() - start
    print "Avg {} sec / doc".format(total/float(n))
    # data_f = 'data/ace_05_yaat.json'

    data_f = 'data/ace_05_head_yaat_train.json'
    print "Writing out data to {}".format(data_f)
    with open(data_f, 'w', encoding='utf8') as outfile:
        outfile.write(unicode(json.dumps(train_annotations, ensure_ascii=False, indent=2)))
    data_f = 'data/ace_05_head_yaat_dev.json'
    print "Writing out data to {}".format(data_f)
    with open(data_f, 'w', encoding='utf8') as outfile:
        outfile.write(unicode(json.dumps(dev_annotations, ensure_ascii=False, indent=2)))
    data_f = 'data/ace_05_head_yaat_test.json'
    print "Writing out data to {}".format(data_f)
    with open(data_f, 'w', encoding='utf8') as outfile:
        outfile.write(unicode(json.dumps(test_annotations, ensure_ascii=False, indent=2)))
