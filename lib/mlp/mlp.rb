module MLP
  class MLP
    def initialize(opts)
      opts[:initial_weight] ||= 0.5
      opts[:learning_rate] ||= 0.2
      opts[:threshold_function] ||= Proc.new { |net| 1 / (1 + Math.exp(-net)) }

      input_nodes = opts[:input]
      #hidden layers = an array of number of nodes per layer, or just a number of layers
      # and we will choose how many nodes per layer?
      hidden_layers = opts[:hidden_layers]
      output_nodes = opts[:output]
      @learning_rate = opts[:learning_rate]
      @threshold_function = opts[:threshold_function]
      initial_weight = opts[:initial_weight]

      #initialize weights matrix
      @weights = []
      #intialize matrix from inputs to hidden layers
      input_to_hidden = []
      number_of_nodes_in_first_hidden_layer = hidden_layers.shift
      input_nodes.times do
        input = []
        number_of_nodes_in_first_hidden_layer.times { input << initial_weight }
        input_to_hidden << input
      end
      @weights << input_to_hidden


      #initialize hidden layers
      hidden_layers.each do |number_of_output_nodes|
        number_of_input_nodes = @weights[-1][0].count
        hidden_layer = []
        number_of_input_nodes.times do
          input = []
          number_of_output_nodes.times { input << initial_weight }
          hidden_layer << input
        end
        @weights << hiddden_layer
      end


      #initialize last hidden layer to output

    end

    # Given an array of inputs and a class, update the weights
    # of the mpl.
    def train(input, class_for_input)
      #compute output - output = input * weight
      intermediate = 0
      @weights.each_with_index do |input, i|
        out
      end
      #how to do this?
    end
  end
end
