module MLP
  class MLP
    def initialize(opts)
      opts[:initial_weight] ||= 0.5
      opts[:learning_rate] ||= 1.0
      opts[:structure] ||= [2, 2, 2]
      opts[:hidden_layers] ||= [2]

      input_nodes = opts[:input]
      #hidden layers = an array of number of nodes per layer, or just a number of layers
      # and we will choose how many nodes per layer?
      # right now it is an array of number of nodes in each layer
      hidden_layers = opts[:hidden_layers]
      output_nodes = opts[:output]
      @learning_rate = opts[:learning_rate]
      @threshold_function = Proc.new { |net| 1 / (1 + Math.exp(-net)) }
      @threshold_function_derivative = Proc.new { |out| out * (1 - out) }
      @initial_weight = opts[:initial_weight]

      @weights = []
      #intialize matrix from inputs to hidden layers
      input_to_hidden = []
      number_of_nodes_in_first_hidden_layer = hidden_layers.shift
      input_nodes.times do
        input = []
        number_of_nodes_in_first_hidden_layer.times { input << @initial_weight }
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
      hidden_layer_to_output = []
      number_of_input_nodes = @weights[-1][0].count
      number_of_input_nodes.times do
        input = []
        output_nodes.times { input << @initial_weight }
        hidden_layer_to_output << input
      end
      @weights << hidden_layer_to_output
    end

    def initialize_activation_nodes
      @activation_nodes = Array.new(@structure.length) do |n|
        Array.new(@structure[n], 1.0)
      end
    end

    #initialize weights matrix
    # @weights is an array of 2d weight matricies
    #         o_1  o_2 o_3 ect
    # input_1
    # input_2
    # input_3
    # ect
    # To get the weight from input node i to output node j of
    # layer l, @weights[l][i][j]
    def initialize_weights
      @weights =  Array.new(@structure.length - 1) do |i|
        origin_nodes = @structure[i]
        target_nodes = @structure[i + 1]
        Array.new(origin_nodes) do |j|
          Array.new(target_nodes) do |k|
            @initial_weight
          end
        end
      end
    end

    def calculate_output(input)
      fill_in_input_later(input)
      @weights.each_index do |i|
        @structure[i + 1].times do |j|
          sum = 0.0
          @activation_nodes[i].each_index do |k|
            sum += (@activation_nodes[i][k] * @weights[i][k][j]
          end
          @activation_nodes[i+1][j] = @threshold_function.call(sum)
        end
      end
    end

    def fill_in_input_later(input)
      input.each_index do |i|
        @activation_nodes.first[i] = input[i]
      end
    end

    def backpropogate(expected_output)
      calculate_error_for_output(expected_output)
      calculate_error_for_hidden_layers
    end

    def calculate_error_for_output(expected_output)
      output_values = @activation_nodes[-1]
      output_error = []
      output_values.each_index do |index|
        output_error << (expected_output[index] - output_values[index]) * @threshold_function_derivative.call(output_values[index])
      end
      @errors = [output_error]
    end

    def calculate_error_for_hidden_layers
      previous_error = @errors[-1]
      (@activation_nodes.size - 2).downto(1) do |n|
        layer_error = []
        @activation_nodes[n].each_index do |i|
          err = 0.0
          @structure[n + 1].times do |j|
            err += previous_error[j] * @weights[n][i][j]
          end
          layer_error[i] = (@threshold_function_derivative.call(@activation_nodes[n][i]) * err)
        end
        previous_error = layer_error
        @errors.unshift(layer_error)
      end
    end

    # Given an array of inputs and a class, update the weights
    # of the mpl.
    #
    # Predicted output of the mlp will be the output node with the
    # highest activation number.
    def get_output_of_each_layer(input)
      #compute output - output = input * weight
      output_of_each_layer = [input]
      @weights.each do |weight_matrix|
        input_to_current_layer = output_of_each_layer[-1]
        output_of_current_layer = Array.new(weight_matrix[0].size, 0.0)
        weight_matrix[0].length.times do |i|
          output_of_node = 0.0
          weight_matrix.each_with_index do |weight, index|
            output_of_node += weight[i] * input_to_current_layer[index]
          end
          output_of_current_layer[i] = @threshold_function.call(output_of_node)
        end
        output_of_each_layer << output_of_current_layer
      end
      return output_of_each_layer
    end

    def train(input, expected_output)
      outputs = get_output_of_each_layer(input)
      # p outputs
      #E_k = (T_k - O_k) * f_prime(net)
      error = []
      error << []
      output_layer = outputs[-1]
      output_layer.each_index do |index|
        error[0] << (expected_output[index] - output_layer[index]) * @threshold_function_derivative.call(output_layer[index])
      end

      outputs[0...-1].zip(@weights).reverse_each do |output_layer, weight_matrix|
        prepend_error_for_hidden_layer(output_layer, weight_matrix, error)
      end

      #now we have all errors. can use to update weights
      @weights = @weights.map do |weight_matrix|
        output = outputs[@weights.index(weight_matrix)]
        error_array = error[@weights.index(weight_matrix) + 1]
        weight_matrix = weight_matrix.map do |input_weight_array|
          input_weight_array.zip(error_array, output).map do |weight_and_error_and_output|
            weight_and_error_and_output[0] +
              (@learning_rate * weight_and_error_and_output[1] * weight_and_error_and_output[2])
          end
        end
      end
      calculate_error(output_layer, expected_output)
    end

    def calculate_error(output, expected_output)
      error = 0.0
      expected_output.each_index do |i|
        error += 0.5 * (output[i] - expected_output[i]) ** 2
      end
      error
    end

    # Given an array of outputs of a hidden layer, and a 2d array of errors
    # for the previous layers, calculate the error for this hidden layer
    # and prepend it to the error array.
    def prepend_error_for_hidden_layer(output_of_hidden_layer, weight_matrix, error_array)
      error = error_array[0]
      error_for_current_layer = []
      #for each output
      output_of_hidden_layer.each_with_index do |output, index|
        # sum of error for output node * weight from input node to output
        sum = weight_matrix[index].zip(error).reduce(0.0) do |sum_so_far, weight_and_error|
          # binding.pry
          sum_so_far += (weight_and_error[0] * weight_and_error[1])
        end
        error_for_current_layer << (sum * @threshold_function_derivative.call(output))
      end
      error_array.unshift(error_for_current_layer)
    end

    # Given a data point as an array of attribute values and an array of classes, return
    # the class predicted by the neural network.
    def get_predicted_class(input, class_array)
      output = get_output_of_each_layer(input)[-1]
      max = output.find_index(output.max)
      class_array[max]
    end
  end
end
