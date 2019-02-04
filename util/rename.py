import tensorflow as tf


def rename_scope(sess, checkpoint_dir, replace_from, replace_to):
  checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
  
  for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
    # Load the variable
    var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

    # Set the new name
    new_name = var_name
    if None not in [replace_from, replace_to]:
      new_name = new_name.replace(replace_from, replace_to)

    # print('Renaming %s to %s.' % (var_name, new_name))
    # Rename the variable
    var = tf.Variable(var, name=new_name)

  # Save the variables
  sess.run(tf.global_variables_initializer())
