import numpy as np

# TODO Substitute the softmax with pytorch implementation
def softmax(mat):
    "calc softmax such that labels per time-step form probability distribution"
    _, maxT = mat.shape
    res = np.zeros(mat.shape)
    for t in range(maxT):
        y = mat[:, t]
        e = np.exp(y)
        s = np.sum(e)
        res[:, t] = e / s
    return res


def greedy_decoder(predicted_matrix_batch, CHARS):
    predicted_labels = list()
    decoded_labels = list()
    for i in range(predicted_matrix_batch.shape[0]):
        predicted_matrix = predicted_matrix_batch[i, :, :]
        predicted_matrix = softmax(predicted_matrix)
        pred_label = list()
        for j in range(predicted_matrix.shape[1]):
            pred_label.append(np.argmax(predicted_matrix[:, j], axis=0))

        # merge repeating labels and blank labels
        no_repeat_blank_label = list()
        k = 0
        while k < len(pred_label) and pred_label[k] == len(CHARS) - 1:
            k += 1
        if k < len(pred_label):
            pre_c = pred_label[k]
            no_repeat_blank_label.append(pre_c)
            for c in pred_label[k + 1:]:
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == len(CHARS) - 1:
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
        predicted_labels.append(no_repeat_blank_label)

    for i, label in enumerate(predicted_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        decoded_labels.append(lb)

    return decoded_labels, predicted_labels


def beam_search_decoder(class_name, predicted_matrix_batch, CHARS, lm=None, beamWidth=25):
    class BeamEntry:
        "information about one single beam at specific time-step"

        def __init__(self):
            self.prTotal = 0  # blank and non-blank
            self.prNonBlank = 0  # non-blank
            self.prBlank = 0  # blank
            self.prText = 1  # LM score
            self.lmApplied = False  # flag if LM was already applied to this beam
            self.labeling = ()  # beam-labeling

    def norm(beam_state):
        "length-normalise LM score"
        for (_, v) in beam_state.items():
            labelingLen = len(v.labeling)
            v.prText = v.prText ** (1.0 / (labelingLen if labelingLen else 1.0))
        return beam_state

    def sort(beam_state):
        "return beam-labelings, sorted by probability"
        beams = [v for (_, v) in beam_state.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal * x.prText)
        return [x.labeling for x in sortedBeams]

    def apply_bigram_lm(parentBeam, childBeam, CHARS, lm):
        "calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars"
        if lm and not childBeam.lmApplied:
            c1 = CHARS[parentBeam.labeling[-1] if parentBeam.labeling else CHARS.index(' ')]  # first char
            c2 = CHARS[childBeam.labeling[-1]]  # second char
            lmFactor = 0.01  # influence of language model
            bigramProb = lm.getCharBigram(c1, c2) ** lmFactor  # probability of seeing first and second char next to each other
            childBeam.prText = parentBeam.prText * bigramProb  # probability of char sequence
            childBeam.lmApplied = True  # only apply LM once per beam entry

    def apply_symbol_mask_lm(parentBeam, childBeam, class_name):
        if not childBeam.lmApplied:
            pos_idx = len(childBeam.labeling)-1
            if class_name == 'KZ':
                letters_pos_idxs = [3, 4, 5]   # position indexes of letters in kz LPs
                numbers_pos_idxs = [0, 1, 2, 6, 7]  # position indexes of numbers in kz LPs
            elif class_name == 'RU':
                letters_pos_idxs = [0, 4, 5]   # position indexes of letters in russian LPs
                numbers_pos_idxs = [1, 2, 3, 6, 7, 8]  # position indexes of numbers in k LPs
            if pos_idx in letters_pos_idxs:
                if childBeam.labeling[-1] >= 10:  # first 10 digits in CHARS are numbers
                    childBeam.prText = parentBeam.prText * 1  # probability of char sequence
                elif childBeam.labeling[-1] < 10:
                    childBeam.prText = parentBeam.prText * 0  # probability of char sequence
            if pos_idx in numbers_pos_idxs:
                if childBeam.labeling[-1] < 10:
                    childBeam.prText = parentBeam.prText * 1  # probability of char sequence
                elif childBeam.labeling[-1] >= 10:
                    childBeam.prText = parentBeam.prText * 0  # probability of char sequence
            childBeam.lmApplied = True  # only apply LM once per beam entry

    "beam search as described by the paper of Hwang et al. and the paper of Graves et al."
    predicted_labels = list()
    decoded_labels = list()
    blankIdx = len(CHARS) - 1
    for i in range(predicted_matrix_batch.shape[0]):
        predicted_matrix = predicted_matrix_batch[i, :, :]
        predicted_matrix = softmax(predicted_matrix)
        maxC, maxT = predicted_matrix.shape

        # initialise beam state
        lastBeamState = {}
        labeling = ()
        lastBeamState[labeling] = BeamEntry()
        lastBeamState[labeling].prBlank = 1
        lastBeamState[labeling].prTotal = 1

        # go over all time-steps
        for t in range(maxT):
            currBeamState = {}

            # get beam-labelings of best beams
            bestLabelings = sort(lastBeamState)[0:beamWidth]

            # go over best beams
            for labeling in bestLabelings:

                # probability of paths ending with a non-blank
                prNonBlank = 0
                # in case of non-empty beam
                if labeling:
                    # probability of paths with repeated last char at the end
                    prNonBlank = lastBeamState[labeling].prNonBlank * predicted_matrix[labeling[-1], t]

                # probability of paths ending with a blank
                prBlank = (lastBeamState[labeling].prTotal) * predicted_matrix[blankIdx, t]

                # add beam at current time-step if needed
                if labeling not in currBeamState:
                    currBeamState[labeling] = BeamEntry()

                # fill in data
                currBeamState[labeling].labeling = labeling
                currBeamState[labeling].prNonBlank += prNonBlank
                currBeamState[labeling].prBlank += prBlank
                currBeamState[labeling].prTotal += prBlank + prNonBlank
                currBeamState[labeling].prText = lastBeamState[labeling].prText  # beam-labeling not changed, therefore also LM score unchanged from
                currBeamState[labeling].lmApplied = True  # LM already applied at previous time-step for this beam-labeling

                # extend current beam-labeling
                for c in range(maxC - 1):
                    # add new char to current beam-labeling
                    newLabeling = labeling + (c,)

                    # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                    if labeling and labeling[-1] == c:
                        prNonBlank = lastBeamState[labeling].prBlank * predicted_matrix[c, t]
                    else:
                        prNonBlank = lastBeamState[labeling].prTotal * predicted_matrix[c, t]

                    # add beam at current time-step if needed
                    if newLabeling not in currBeamState:
                        currBeamState[newLabeling] = BeamEntry()

                    # fill in data
                    currBeamState[newLabeling].labeling = newLabeling
                    currBeamState[newLabeling].prNonBlank += prNonBlank
                    currBeamState[newLabeling].prTotal += prNonBlank

                    # apply LM
                    apply_symbol_mask_lm(currBeamState[labeling], currBeamState[newLabeling], class_name)
                    # if lm:
                    #     apply_bigram_lm(currBeamState[labeling], currBeamState[newLabeling], CHARS, lm)

            # set new beam state
            lastBeamState = currBeamState

        # normalise LM scores according to beam-labeling-length
        lastBeamState = norm(lastBeamState)

        # sort by probability
        bestLabeling = sort(lastBeamState)[0]  # get most probable labeling
        predicted_labels.append(bestLabeling)

    # map labels to chars
    for labels in predicted_labels:
        res = ''
        for l in labels:
            res += CHARS[l]
        decoded_labels.append(res)

    return decoded_labels, predicted_labels
