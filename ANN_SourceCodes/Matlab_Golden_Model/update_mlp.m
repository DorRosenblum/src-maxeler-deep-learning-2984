function [model] = update_mlp(model, input, target)
% this function is the Back Propogation, it is called once for EVERY Entry (pattern presentation), weights are
% updated every time, which is the magical step. (ie its not a batch update, but a per-entry)

% this holds the activation of every neuron in every layer
    activations = cell(length(model.weights)+1,1);
    activations{1} = input;
    
    if (model.disable_export == 0)
        dlmwrite('csv files/training_set.csv'     ,input,      '-append');
        dlmwrite('csv files/training_class.csv'   ,target,     '-append');
    end
    
    % SAME CODE LIKE "test_mlp" - but saves each layers result insted of
    % just the last layer's output from one input entry
    % this loop calculates the activations of all the neuron layers
    for i = 1:length(model.weights)
        % activations{i} is a row vector
        % model.weights{i} is a matrix of weights
        % the output of that product is a row vector of length equal to the
        % number of neurons in the next layer
        temp = activations{i} * model.weights{i} + model.biases{i};     % temp = layer CB
        
        if ((i == length(model.weights)) && (model.use_softmax_for_final_layer == 1))
            activations{i+1} = activetion_function_sel(temp, 'SoftMax INT');
        else    %use normal activetion
            activations{i+1} = activetion_function_sel(temp, model.actietion_func_sel);%1./(1+exp(-((temp)))); % squash the output a bit
        end
    
        % export the model's initial_model_weights&biases:
        if (model.disable_export == 0)
            dlmwrite('csv files/CB.csv',          temp,'-append');
            dlmwrite('csv files/activations.csv', activations{i+1} ,'-append');
        end
    end

    
    
    
    % variable for holding the errors at each level
    errors = cell(length(model.weights),1);
    weights_delta = cell(length(model.weights),1);

            
    % this code propagates the error back through the neural net
    run_error = (target - activations{end}); %%%%%% 0.5.*(target - activations{end}).^2;
    %keeps track of the error at each loop
    for i = length(model.weights):-1:1
        %this error per net is correct only for Sigmond Activation function
        %that holds dout/dnet = activations .* (1-activations)
        if ((i == length(model.weights)) && (model.use_softmax_for_final_layer == 1))
            errors{i}           = (run_error) .* derivative_activetion_function_sel([],activations{i+1},'SoftMax');                 %activations{i+1} .* (1-activations{i+1}) .* (run_error);% derror/dweight = dout/dnet * run_error
        else    %use normal activetion
            errors{i}           = (run_error) .* derivative_activetion_function_sel([],activations{i+1},model.actietion_func_sel);
        end

        weights_delta{i}    = activations{i}' * errors{i};  % the total delte of weight by what acctually activations outputted
        if (model.batch_mode_update == 1)
            model.batch_sum_weights_delta{i}    = model.batch_sum_weights_delta{i} + weights_delta{i};
            model.batch_sum_biases_delta{i}     = model.batch_sum_biases_delta{i}  + errors{i};
        end
        run_error           = errors{i} * model.weights{i}';  % pass back each net error through appropriate Connectivety Band
        
        % export the model's initial_model_weights&biases:
        if (model.disable_export == 0)
            dlmwrite('csv files/mirror_BP.csv',   run_error,'-append');
            dlmwrite('csv files/mirror_BP_CB.csv',errors{i} ,'-append');
        end
    end
    
    
    
    % this code updates the weights and biases
    if (model.batch_mode_update == 0)
        
        for i = 1:length(model.weights)
            % update weights based on the learning rate, the input activation
            % and the error
            model.weights{i} = model.weights{i} + model.learning_rate .* weights_delta{i}; %activations{i}' * errors{i};
            
            
            % update the neuron biases as well
            if (disable_bias == 1)
                model.biases{i} =  zeros(size(model.biases{i}));
            else
                model.biases{i} =  model.biases{i} + model.learning_rate * errors{i};
            end
            % it takes a while to figure out all the matrix operations, but
            % once it's done it's nice.
        end
    end
    
end