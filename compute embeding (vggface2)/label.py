# user/bin/python3
import pickle
from embeding_vgg import *


MIN_THRESHOLD = 0.7
MAX_THRESHOLD = 1.9
model_dir ='compute embeding (vggface2)/vgg_model/20180402-114759/'
image_batch = 500
image_size = 160
image_files = 'compute embeding (vggface2)/296'
criterion_image_path = 'compute embeding (vggface2)/images/'


def embeding(image_path):

    # fetch the classes (labels as strings) exactly as it's done in get_dataset
    image_list, label_list = get_dataset(image_path)

    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            load_model(model_dir)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            nrof_images = len(image_list)
            print('Number of images: ', nrof_images)
            batch_size = image_batch
            if nrof_images % batch_size == 0:
                nrof_batches = nrof_images // batch_size

            else:
                nrof_batches = (nrof_images // batch_size) + 1

            print('Number of batches: ', nrof_batches)
            embedding_size = embeddings.get_shape()[1]
            emb_array = np.zeros((nrof_images, embedding_size))
            start_time = time.time()

            for i in range(nrof_batches):
                if i == nrof_batches -1:
                    n = nrof_images

                else:
                    n = i*batch_size + batch_size

                # Get images for the batch
                images = load_data(image_list[i*batch_size:n], False, False, image_size)

                feed_dict = { images_placeholder: images, phase_train_placeholder:False }

                # Use the facenet model to calcualte embeddings
                embed = sess.run(embeddings, feed_dict=feed_dict)
                emb_array[i*batch_size:n, :] = embed
                print('Completed batch', i+1, 'of', nrof_batches)

            run_time = time.time() - start_time
            print('Run time: ', run_time)

            #export emedings and labels and save it with pickle format
            data = {"embeddings": embed, "names": label_list}
            return data

def save_label():
    criterion_image_embedings = embeding(criterion_image_path)
    foo = embeding(image_files)
    diff_embeding = np.zeros([
        len(foo['names']),
        len(foo['embeddings'][0])
    ])

    labels = []
    names = []
    for i, img_embeding in enumerate(criterion_image_embedings['embeddings']):
        for j, embeding_ in enumerate(foo['embeddings']):
            diff_embeding[j] = np.subtract(img_embeding, embeding_)

        mean = np.mean(diff_embeding)
        if mean < MIN_THRESHOLD:
            label = 1

        elif MAX_THRESHOLD < mean < MIN_THRESHOLD:
            label = 2

        else:
            label = 3

        labels.append(label)
        names.append(criterion_image_embedings['names'][i])

    data = {"names": names, 'labels':labels}

if __name__ == '__main__':
    save_label()
