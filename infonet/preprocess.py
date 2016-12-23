# all mapping schemes assume the annotation spans correspond with python indexing
# eg ann-span = [0,2] means the interval [0,2) or tokens[0:2]

def Entity_BIO_map(mention_labels, annotation):
    """ Uses BIO scheme (untyped) for entities only """
    if annotation['node-type'] == 'entity':
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
    # make sure there are no overlapping mentions
    # NOTE: There are 3 overlapping mentions in the dataset
    #   They are all due to tokenization errors for words that
    #   are not split on '/'s, eg 'Secretary/Advisor'
    #   Since there are only 3, we just accept and
    #   ignore those sources of error.
    # def is_bad_overlap(x,y):
    #     # can't bad overlap if either boundaries are same
    #     # case: [ )   or  [ )
    #     #       [  )     [  )
    #     if (x[0] == y[0]) or (x[1] == y[1]):
    #         return False
    #     # only check from left to right
    #     l, r = (x,y) if x[0] < y[0] else (y,x)
    #     # case [  )    (what we're looking for)
    #     #        [  )
    #     if l[1] > r[0] and l[1] < r[1]:
    #         return True
    #     # case [   )  (true nesting is ok)
    #     #        [)
    #     else:
    #         return False

    # for i, a in enumerate(annotations):
    #     for b in annotations[i+1:]:
    #         assert not is_bad_overlap(a['ann-span'], b['ann-span']), (a,b)
    # assert not any([ is_bad_overlap(a['ann-span'], b['ann-span'])
    #                  for i, a in enumerate(annotations)
    #                  for b in annotations[i+1:] ])

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
        'type_map':{t:t.split('-')[1]
                    if len(t.split('-')) > 1
                    else None
                    for t in boundary_vocab.vocabset
                    }
    }
    return tag_map

def compute_mentions(doc, fine_grained=False):
    """ Compute gold mentions as (*span, type) from yaat.

    Return them sorted lexicographicaly. """
    mentions = []
    for ann in doc['annotations']:
        if ann['ann-type'] == u'node':
            mention_type = ann['node-type']+':'+ann['type']
            if fine_grained:
                mention_type += ':'+ann['subtype']
            mentions.append((ann['ann-span'][0], ann['ann-span'][1], mention_type))
    return sorted(mentions, key=lambda x: x[:2])

def compute_relations(doc, fine_grained=False):
    """ Compute gold relations as (*left_span, *right_span, type) from yaat """
    relations = []
    id2ann = { ann['ann-uid']:ann for ann in doc['annotations']}
    for ann in doc['annotations']:
        if ann['ann-type'] == u'edge':
            left_span = id2ann[ann['ann-left']]['ann-span']
            right_span = id2ann[ann['ann-right']]['ann-span']
            rel_type = ann['edge-type']+':'+ann['type']
            if fine_grained:
                rel_type += ':'+ann['subtype']
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
        for qnode in resolved_annotations:
            if not overlaps(node, qnode):
                resolved_annotations.append(node)

    # now only include relations whose constiuent mentions are still around
    node_id_set = set([node['ann-uid'] for node in resolved_annotations ])
    for ann in annotations:
        if ann['ann-type'] == u'edge':
            if ann['ann-left'] in node_id_set and ann['ann-right'] in node_id_set:
                resolved_annotations.append(ann)
    return resolved_annotations
