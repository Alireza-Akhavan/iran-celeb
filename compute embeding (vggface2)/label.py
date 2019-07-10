# user/bin/python3
import pickle
from embeding_vgg import *


MIN_THRESHOLD = 0.7
MAX_THRESHOLD = 1.9
model_dir ='vgg_model/20180402-114759/'
image_batch = 500
image_size = 160
pickle_filename = 'embeddings.pickle'
criterion_image_path = 'images/'


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
    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    criterion_image_embedings = embeding(criterion_image_path)
    import pudb; pudb.set_trace()  # XXX BREAKPOINT

    foo = pickle.load(open(pickle_filename,'rb'))
    for i in foo['embeddings']:
        print(i)

if __name__ == '__main__':
    save_label()
