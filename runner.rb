require_relative 'lib/mlp'
require 'ai4r'

# hrrm = Ai4r::NeuralNetwork::Backpropagation.new([2, 2, 2])
# hrrm.set_parameters({ disable_bias: true, momentum: 0, initial_weight_function: Proc.new { |a, c, b| 0.5 } })
# 3000.times do
#   hrrm.train([0, 0], [1, 0])
#   hrrm.train([1, 1], [0, 1])
#   hrrm.train([1, 0],  [1, 0])
#   hrrm.train([1, 1],  [0, 1])
#   hrrm.train([1, 1],  [0, 1])
#   hrrm.train([0, 1],  [1, 0])
#   hrrm.train([1, 1],  [0, 1])

#   hrrm.train([1, 1],  [0, 1])
#   hrrm.train([1, 0],  [1, 0])
#   hrrm.train([0, 0],  [1, 0])
#   hrrm.train([1, 1],  [0, 1])
#   hrrm.train([0, 1],  [1, 0])
#   hrrm.train([1, 1],  [0, 1])
# end


mlp = MLP::MLP.new({ input: 2, output: 2, learning_rate: 0.25 })
3000.times do
  mlp.train([0, 0], [1, 0])
  mlp.train([1, 1], [0, 1])
  mlp.train([1, 0],  [1, 0])
  mlp.train([1, 1],  [0, 1])
  mlp.train([1, 1],  [0, 1])
  mlp.train([0, 1],  [1, 0])
  mlp.train([1, 1],  [0, 1])

  p mlp.train([1, 1],  [0, 1])
  mlp.train([1, 0],  [1, 0])
  mlp.train([0, 0],  [1, 0])
  mlp.train([1, 1],  [0, 1])
  mlp.train([0, 1],  [1, 0])
  mlp.train([1, 1],  [0, 1])
end
