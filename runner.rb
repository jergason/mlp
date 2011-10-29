require_relative 'lib/mlp'
require 'woof'
require 'pry'
require 'csv'

def make_to_input_and_output_arrays(arff_file)
  output_length = arff_file.get_class_values.length
  arff_file.data.map do |data_instance|
    # class_attribute = data_instance.fetch("class")
    class_attribute = data_instance.fetch(arff_file.class_attribute)
    # binding.pry
    inputs = data_instance.map do |key, value|
      value
    end
    # count = 0
    # inputs.each do |inp|
    #   count += 1 if inp == data_instance.fetch("class")
    # end
    # binding.pry if count > 1
    # Assume that the last value is always the class, for now
    inputs.pop
    outputs = arff_file.get_class_values.map do |value|
      value == arff_file.get_class_values[class_attribute] ? 1.0 : 0.0
    end
    [inputs, outputs]
  end
end

def make_new_mlp(opts)
  opts[:learning_rate] ||= 0.1
  opts[:structure] ||= [4, 4, 4]
  return MLP::MLP.new(opts)
end

def test_accuracy(mlp, validation_set)
  correct, wrong = 0, 0
  validation_input_and_output = make_to_input_and_output_arrays(validation_set)
  validation_input_and_output.each do |input, target_output|
    prediction = mlp.calculate_output(input)
    predicted_class = prediction.find_index(prediction.max)
    actual_class = target_output.find_index(target_output.max)
    if predicted_class == actual_class
      correct += 1
    else
      wrong += 1
    end
  end
  [correct, wrong]
end

def train_over_time(training_set, validation_set, mlp=nil, iterations=10, times_per_iteration=100)
  results = []
  training_in_and_out = make_to_input_and_output_arrays(training_set)
  if mlp.nil?
    mlp = MLP::MLP.new({
      :structure => [training_set.data[0].count - 1, 6, 4, training_set.get_class_values.length],
      :learning_rate => 0.9
    })
  end
  iterations.times do |i|
    times_per_iteration.times do |j|
      training_in_and_out.each do |input, output|
        mlp.train(input, output)
      end
    end

    correct, incorrect = test_accuracy(mlp, validation_set)
    results << { :iterations => (i + 1) * times_per_iteration, :accuracy => correct.to_f / (correct + incorrect) }
  end
  results
end

def test_mlp(training_set, validation_set, type_of_test, initial_value, increment_by, num_iterations)
  results = []
  # binding.pry
  training_in_and_out = make_to_input_and_output_arrays(training_set)
  num_iterations.times do |i|
    if type_of_test == :learning_rate
      mlp = make_new_mlp({ :learning_rate => initial_value, :structure => [training_set.data[0].count - 1, 5, training_set.get_class_values.length]})
    elsif type_of_test == :hidden_nodes
      mlp = make_new_mlp({ :learning_rate => 0.7, :structure => [training_set.data[0].count - 1, initial_value, training_set.get_class_values.length]})
    end
    500.times do
      training_in_and_out.each do |input, output|
        mlp.train(input, output)
      end
    end

    # now test accuracy
    correct, wrong = test_accuracy(mlp, validation_set)
    # correct, wrong = 0, 0
    # validation_input_and_output = make_to_input_and_output_arrays(validation_set)
    # validation_input_and_output.each do |input, target_output|
    #   prediction = mlp.calculate_output(input)
    #   predicted_class = prediction.find_index(prediction.max)
    #   actual_class = target_output.find_index(target_output.max)
    #   if predicted_class == actual_class
    #     correct += 1
    #   else
    #     wrong += 1
    #   end
    # end
    # binding.pry
    if type_of_test == :learning_rate
      results << { :learning_rate => initial_value, :accuracy => (correct.to_f / (correct + wrong).to_f) }
    elsif type_of_test == :hidden_nodes
      results << { :hidden_nodes => initial_value, :accuracy => (correct.to_f / (correct + wrong).to_f) }
    end
    initial_value += increment_by
  end
  results
end


def two_layer_vowel
  file = Woof::Parser.new('data/vowel.arff').parse
  file.remove_attribute("'Train or Test'")
  file.continuize!
  sets = file.get_training_and_validation_sets(0.75, 0.25)
  results = train_over_time(sets[0], sets[1])

  iterations = []
  accuracies = []
  results.each do |res|
    iterations << res[:iterations]
    accuracies << res[:accuracy]
  end

  puts "two_layer_iterations <- c(#{iterations.join(",")})"
  puts "two_layer_accuracies <- c(#{accuracies.join(",")})"
end



def momentum(arff_file, training_split, validation_split)
  arff_file.continuize!
  sets = arff_file.get_training_and_validation_sets(training_split, validation_split)
  results = train_over_time(sets[0], sets[1], MLP::MLP.new({ :structure => [sets[0].data[0].length - 1, 13, sets[0].get_class_values.length], :learning_rate => 0.75, :momentum => 0.00001 }))

  iterations = []
  accuracies = []
  results.each do |res|
    iterations << res[:iterations]
    accuracies << res[:accuracy]
  end

  puts "momentum_iterations <- c(#{iterations.join(",")})"
  puts "momentum_accuracies <- c(#{accuracies.join(",")})"
end

vowel = Woof::Parser.new('data/vowel.arff').parse
vowel.remove_attribute("'Train or Test'")
momentum(vowel, 0.75, 0.25)
iris = Woof::Parser.new('data/iris.arff').parse
momentum(iris, 0.7, 0.3)
# sets = file.get_training_and_validation_sets(0.7, 0.3)
# learning_rate_tests = test_mlp(sets[0], sets[1], :learning_rate, 0.1, 0.1, 10)
# node_number_tests = test_mlp(sets[0], sets[1], :hidden_nodes, 1, 1, 15)

# rates = []
# accuracies = []
# learning_rate_tests.each do |test|
#   rates << test[:learning_rate]
#   accuracies << test[:accuracy]
# end

# puts "rates <- c(#{rates.join(",")})"
# puts "rate_accuracies <- c(#{accuracies.join(",")})"

# node_number = []
# node_accuracy = []
# node_number_tests.each do |test|
#   node_number << test[:hidden_nodes]
#   node_accuracy << test[:accuracy]
# end

# puts "node_numbers <- c(#{node_number.join(",")})"
# puts "node_accuracies <- c(#{node_accuracy.join(",")})"
