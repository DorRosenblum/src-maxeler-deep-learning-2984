function [model cc output] = train_mlp(input, target,...
                                        batch_mode_update,epoch_size,load_model_from_file,disable_bias,disable_export,...
                                        hidden,  iterations, learning_rate, momentum, actietion_func_sel,use_softmax_for_final_layer)
    % this is the function that handles all the looping and running of the
    % neural network, it initializes the network based on the number of
    % hidden layers, and presents every item in the input over and over,
    % $iterations$ times.
    % hidden is the only complicated variables.  Like weka, it accepts a
    % list of values (as a row vector), and interprets it as the number of
    % neurons in each hidden layer, so [2 2 2] means there will be an input
    % layer defined by the size of the input, three hidden layers, with 2 
    % neurons each, and the output layer, defined by the target.  I think
    % it's pretty good.
    

    
    % characterize the input and output
    [ntrain nInLayer]   = size(input);    % ntrain== number of data entrys, nInLayer== input data vec size
    [jnk nOutLayer]     = size(target);

    % keep track of how many neurons are in each layer
    nNeurons = [nInLayer hidden nOutLayer];
    nNeurons(nNeurons == 0) = []; % remove 0 layers, to allow putting a zero for no hidden layers

    % there are one fewer sets of weights and biases as total layers
    nTransitions = length(nNeurons)-1;      % num of wights is -1 for being in between layers

    
    
    
    if (load_model_from_file ~= 1)
        % initialize the output
        model = [];
       
        % initialize the weights between layers, and the biases: 
        for i = 1:nTransitions % initialize the weights between layers, and the biases (past the first layer)
            model.weights{i} = randn(nNeurons(i),nNeurons(i+1)); 
            % the weight matrix has X rows, where X is the number of input
            % neurons to the layer, and Y columns, where Y is the number of
            % output neurons.  multiplication of the input with the weight
            % matrix transforms the dimensionality of the input to that of the
            % output.  Initialization is done here randomly.
            if (disable_bias == 1)
                model.biases{i} = zeros(1,nNeurons(i+1));
            else
                model.biases{i} = randn(1,nNeurons(i+1));
            end
            % biases are random as well

            
            % export the model's initial_model_weights&biases:
            dlmwrite('csv files/initial_model_weights.csv',   model.weights{i},'-append');
            dlmwrite('csv files/initial_model_biases.csv',    model.biases{i},'-append');
                    
            
        %I tihnk its not in use... %model.lastdelta{i} = 0;  
        end
    else
        load('model.mat','model');
    end
    model.learning_rate     = learning_rate;
    model.momentum          = momentum;          % NOT IN USE CURERNTLY!!! % for some heavy ball action
    model.batch_mode_update = batch_mode_update;
    model.epoch_size        = epoch_size;
    model.disable_bias      = disable_bias;
    model.disable_export    = disable_export;
    model.actietion_func_sel= actietion_func_sel;
    model.use_softmax_for_final_layer = use_softmax_for_final_layer;
    function init_epochs_batch_sum_weights()
        % init epoch's batch sums
        if (model.batch_mode_update == 1)
            %%% init model.batch_sum_weights_delta to zero
            model.batch_sum_weights_delta = cell(length(model.weights),1);
            for i = 1:length(model.batch_sum_weights_delta)
                model.batch_sum_weights_delta{i} = zeros(size(model.weights{i}));
            end

            model.batch_sum_biases_delta = cell(length(model.biases),1);
            for i = 1:length(model.batch_sum_biases_delta)
                model.batch_sum_biases_delta{i} = zeros(size(model.biases{i}));
            end
        end
    end
    

    %training same set for iterations times, each time perm order differs:
    for i = 1:iterations % repeat the whole training set over and over
        order = randperm(ntrain);  % randomize the presentations % so the access is a diffrant permutation each func run
        
        init_epochs_batch_sum_weights();

        % update weights
        for j = 1:ntrain
            % update_mlp is where the training is actually done PER SINGLE DATA ENRTY
            model = update_mlp(model, input(order(j),:), target(order(j),:)); %PER SINGLE DATA ENRTY
            
            
            % at the end on the epoch, uptate the batch weights and biases
            if ((model.batch_mode_update == 1) && (mod(j,model.epoch_size)==0))
                for i = 1:length(model.weights)
                    model.weights{i} = model.weights{i} + (model.learning_rate./model.epoch_size ).* model.batch_sum_weights_delta{i};
                   
                  
                    if (disable_bias == 1)
                        model.biases{i}     = zeros(size(model.biases{i}));
                    else
                        model.biases{i}     = model.biases{i}  + (model.learning_rate./model.epoch_size ).* model.batch_sum_biases_delta{i};
                    end
                    
                    
                    % export the model's initial_model_weights&biases:
                    if (disable_export == 0)
                        dlmwrite('csv files/new_model_weights.csv',   model.weights{i},'-append');
                        dlmwrite('csv files/new_model_biases.csv',    model.biases{i},'-append');
                    end
                end
                
                init_epochs_batch_sum_weights();
            end
        end
        % for exporting small files for one iterations:
        model.disable_export    =   1;
    end
    
    
   
    
    
    

    
    
    
    
    % test the performance on the training set AFTER all is trained.
    [output cc] = test_mlp(model, input, target);
end
