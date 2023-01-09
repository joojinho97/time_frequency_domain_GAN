
import tensorflow as tf
def generator_loss(disc_generated_output, gen_output, target,p):

    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss =loss_object(tf.ones_like(disc_generated_output), disc_generated_output)


    l1_loss = tf.abs(target - gen_output)
    
    l1_loss = tf.reduce_mean(l1_loss)
    
    
    total_gen_loss = l1_loss*50 + gan_loss

    
    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  
    total_disc_loss = real_loss + generated_loss
  
    return total_disc_loss
    
class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, initial_learning_rate):
    self.initial_learning_rate = initial_learning_rate

  def __call__(self, step):
    if (step < int(1416*5)) :
      return self.initial_learning_rate 
    elif step %int(1416)==0:
      return self.initial_learning_rate*0.95
    else:
      return self.initial_learning_rate
