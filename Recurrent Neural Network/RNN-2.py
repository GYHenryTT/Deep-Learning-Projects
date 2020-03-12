import numpy as np
import tensorflow as tf
import collections
import re

start_token = 'G'
end_token = 'E'
batch_size = 16


def process_poems(file_name):
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        for line in f.readlines():
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content or \
                                start_token in content or end_token in content:
                    continue
                if len(content) < 5 or len(content) > 80:
                    continue
                content = start_token + content + end_token
                poems.append(content)
            except ValueError as e:
                pass
    # Order by word counts of lines
    poems = sorted(poems, key=lambda line: len(line))
    # Collect word counts
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # Collect word counts and frequencies
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # Sort
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


def process_tangshi(file_name):
    poems = []
    with open(file_name, "r", encoding='utf-8', ) as f:
        content = ''
        for line in f.readlines():
            try:
                match = re.match('\n', line)
                if start_token in content or end_token in content:
                    continue
                # if len(line) < 5 or len(line) > 80:
                #     continue
                if match is not None:
                    content = start_token + content + end_token
                    poems.append(content)
                    content = ''
                    continue
                content += line
                content = content.strip()
            except ValueError as e:
                pass

    # Order by word counts of lines
    poems = sorted(poems, key=lambda line: len(line))
    # Collect word counts
    all_words = []
    for poem in poems:
        all_words += [word for word in poem]
    counter = collections.Counter(all_words)  # Collect word counts and frequencies
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])  # Sort
    words, _ = zip(*count_pairs)
    words = words[:len(words)] + (' ',)
    word_int_map = dict(zip(words, range(len(words))))
    poems_vector = [list(map(word_int_map.get, poem)) for poem in poems]
    return poems_vector, word_int_map, words


def merge(vocabulary1, vocabulary2, word_int_map):
    for w in vocabulary2:
        if w not in vocabulary1:
            vocabulary1 += (w,)
            word_int_map[w] = len(word_int_map)
    return vocabulary1, word_int_map


def rnn_model(model, input_data, output_data, vocab_size, rnn_size=128, num_layers=2, batch_size=16,
              learning_rate=0.01):
    end_points = {}
    # Create RNN cell
    if model == 'rnn':
        cell_fun = tf.contrib.rnn.BasicRNNCell
    elif model == 'gru':
        cell_fun = tf.contrib.rnn.GRUCell
    else:
        cell_fun = tf.contrib.rnn.BasicLSTMCell
    
    # Two layers, each layer has 128 RNN cells. output Ht and Ct are in separate tuples
    cell = cell_fun(rnn_size, state_is_tuple=True)
    cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
    
    # Training mode and generating mode (it is generating mode if output is None)
    if output_data is not None:
        initial_state = cell.zero_state(batch_size, tf.float32)
    else:
        initial_state = cell.zero_state(1, tf.float32)

    # Build hidden layers
    with tf.device("/cpu:0"):
        embedding = tf.Variable(tf.random_uniform([vocab_size + 1, rnn_size], -1.0, 1.0), name='embedding')
        inputs = tf.nn.embedding_lookup(embedding, input_data)
        
    outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
    output = tf.reshape(outputs, [-1, rnn_size])

    weights = tf.Variable(tf.truncated_normal([rnn_size, vocab_size + 1]))
    bias = tf.Variable(tf.zeros(shape=[vocab_size + 1]))
    logits = tf.nn.bias_add(tf.matmul(output, weights), bias=bias)  # 一层全连接

    if output_data is not None:  # Training mode
        labels = tf.one_hot(tf.reshape(output_data, [-1]), depth=vocab_size + 1)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        total_loss = tf.reduce_mean(loss)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)  # 优化器用的 adam
        end_points['initial_state'] = initial_state
        end_points['output'] = output
        end_points['train_op'] = train_op
        end_points['total_loss'] = total_loss
        end_points['loss'] = loss
        end_points['last_state'] = last_state
    else:  # Generating mode
        prediction = tf.nn.softmax(logits)
        end_points['initial_state'] = initial_state
        end_points['last_state'] = last_state
        end_points['prediction'] = prediction
    return end_points


def run_training():
    # Process dataset
    poems_vector, word_to_int, vocabularies1 = process_poems(
        './poems.txt')
    tangshi_vector, _, vocabularies2 = process_tangshi(
        './tangshi.txt')
    vocabularies, word_to_int = merge(vocabularies1, vocabularies2, word_to_int)
    # Generate batches
    batches_inputs, batches_outputs = generate_batch(16, tangshi_vector, word_to_int)
    input_data = tf.placeholder(tf.int32, [batch_size, None])
    output_targets = tf.placeholder(tf.int32, [batch_size, None])
    # Build model
    end_points = rnn_model(model='lstm', input_data=input_data, output_data=output_targets, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=16, learning_rate=0.01)

    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        saver.restore(sess, './poem_generator')
        print('restore finished')
        sess.run(init_op)
        for epoch in range(20):
            n = 0
            n_chunk = len(tangshi_vector) // batch_size
            for batch in range(n_chunk):
                print(len(batches_inputs[n]))
                print(len(batches_outputs[n]))
                loss, _, _ = sess.run([
                    end_points['total_loss'],
                    end_points['last_state'],
                    end_points['train_op']
                ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                n += 1
                print('[INFO] Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))
        saver.save(sess, './RNN2')


def gen_poem(begin_word):
    batch_size = 1
    poems_vector, word_to_int, vocabularies1 = process_poems('./poems.txt')
    tangshi_vector, _, vocabularies2 = process_tangshi('./tangshi.txt')
    vocabularies, word_int_map = merge(vocabularies1, vocabularies2, word_to_int)

    input_data = tf.placeholder(tf.int32, [batch_size, None])

    end_points = rnn_model(model='lstm', input_data=input_data, output_data=None, vocab_size=len(
        vocabularies), rnn_size=128, num_layers=2, batch_size=16, learning_rate=0.01)
        
    # assign a begining word
    word = begin_word

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, './RNN2')  # Reload model
        print("restore finished")
        poem = start_token
        # Generate poem start with the begining word
        while word != ' ' and word != 'E':
            poem += word
            word_input = np.array([[word_int_map[w] for w in poem]], dtype=np.int32)
            prediction = sess.run(end_points['prediction'], feed_dict={input_data: word_input})
            word = to_word(prediction[-1], vocabularies)

    return poem[1:]


def generate_batch(batch_size, poems_vec, word_to_int):
    # Each training batch contains 16 poetries
    n_chunk = len(poems_vec) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = poems_vec[start_index:end_index]
        # Find the longest poem in this batch
        length = max(map(len, batches))
        # Fill an empty batch with the index of ' '
        x_data = np.full((batch_size, length), word_to_int[' '], np.int32)
        for row in range(batch_size):
            x_data[row, :len(batches[row])] = batches[row]
        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]
        """
        x_data             y_data
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(x_data)
        y_batches.append(y_data)
    return x_batches, y_batches


def to_word(predict, vocabs):  # Convert results to Chinese words
    sample = np.argmax(predict)
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]


def pretty_print_poem(poem):  # Make a pretty result
    poem_sentences = poem.split('。')
    for s in poem_sentences:
        if s != '' and len(s) > 10:
            print(s + '。')


tf.reset_default_graph()
print('[INFO] train tang poem...')
run_training()  # Train model
print('[INFO] write tang poem...')
poem_keyword = np.array(['日', '红', '山', '夜', '湖', '海', '月'])
for keyword in poem_keyword:
    tf.reset_default_graph()
    poem2 = gen_poem(keyword) # Generate poem
    print("#" * 25)
    pretty_print_poem(poem2)
    print('#' * 25)