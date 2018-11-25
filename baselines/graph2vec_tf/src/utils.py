import os,json


def get_files(dirname, extn, max_files=0):
    all_files = [os.path.join(dirname, f) for f in os.listdir(dirname) if f.endswith(extn)]
    for root, dirs, files in os.walk(dirname):
        for f in files:
            if f.endswith(extn):
                all_files.append(os.path.join(root, f))

    all_files = list(set(all_files))
    all_files.sort()
    if max_files:
        return all_files[:max_files]
    else:
        return all_files


def save_graph_embeddings(corpus, final_embeddings, opfname):
    dict_to_save = {}
    for i in range(len(final_embeddings)):
        graph_fname = corpus._id_to_graph_name_map[i]
        graph_embedding = final_embeddings[i,:].tolist()
        dict_to_save[graph_fname] = graph_embedding

    with open(opfname, 'w') as fh:
        json.dump(dict_to_save,fh,indent=4)


def get_class_labels(graph_files, class_labels_fname):
    graph_to_class_label_map = {l.split()[0].split('.')[0]: int(l.split()[1].strip()) for l in open (class_labels_fname)}
    labels = [graph_to_class_label_map[os.path.basename(g).split('.')[0]] for g in graph_files]

    return labels

if __name__ == '__main__':
    print('nothing to do')
