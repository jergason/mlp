module MLP
  class MLP
    def initialize(opts)
      opts[:initial_weight] ||= 0.5
      opts[:learning_rate] ||= 1.0
      opts[:structure] ||= [2, 2, 2]

      @structure = opts[:structure]
      @learning_rate = opts[:learning_rate]
      @threshold_function = Proc.new { |net| 1 / (1 + Math.exp(-net)) }
      @threshold_function_derivative = Proc.new { |out| out * (1 - out) }
      @initial_weight = opts[:initial_weight]

      initialize_activation_nodes
      initialize_weights
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
      fill_in_input_layer(input)
      @weights.each_index do |i|
        @structure[i + 1].times do |j|
          sum = 0.0
          @activation_nodes[i].each_index do |k|
            sum += (@activation_nodes[i][k] * @weights[i][k][j])
          end
          @activation_nodes[i+1][j] = @threshold_function.call(sum)
        end
      end
      @activation_nodes.last
    end

    def fill_in_input_layer(input)
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

    def update_weights_from_error
      (@weights.length - 1).downto(0) do |n|
        @weights[n].each_index do |i|
          @weights[n][i].each_index do |j|
            delta_w = @activation_nodes[n][i] * @errors[n][j]
            @weights[n][i][j] += @learning_rate * delta_w
          end
        end
      end
    end

    def train(input, expected_output)
      output = calculate_output(input)
      backpropogate(expected_output)
      update_weights_from_error
      calculate_error(output, expected_output)
    end

    def calculate_error(output, expected_output)
      error = 0.0
      expected_output.each_index do |i|
        error += 0.5 * (output[i] - expected_output[i]) ** 2
      end
      error
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
