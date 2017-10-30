function [output cc] = test_mlp(model, input, target)
    % this function is the ANN's Feed Forward.
    % it calculates the output of the model, and compares it to the
    % target output via Cross-correlation
    [ntest nOutLayer] = size(target);
    
    % this variable is for the final output of the neural net
    output = zeros(ntest, nOutLayer);
    for i = 1:ntest         % per data entry:
        temp = input(i,:);  % output at each layer, gets updated
        % temp  initially is a single input data enrty vec
        for j = 1:length(model.weights)
            % pass the j net layer trough a connectivety band of wights and and baies
            % and overwtie it into temp, remember its a matric * operation:
            temp = temp * model.weights{j} + model.biases{j}; % calculate the CB layer
            
            % after passing the temp's layer data through the output of
            % connectivety band, do the activation neuron function on
            % entire layer vec
            if ((j == length(model.weights)) && (model.use_softmax_for_final_layer == 1))
                temp = activetion_function_sel(temp, 'SoftMax INT');%1./(1+exp(-(temp))); % squashit % sigmond!
            else    %use normal activetion
                temp = activetion_function_sel(temp, model.actietion_func_sel);%1./(1+exp(-(temp))); % squashit % sigmond!
            end

        end
        output(i,:) = temp; % keep only the last output value for it is the last layer's output
    end
    
    % compares out to the target output via Cross-correlation
    warning('off', 'all') % corrcoef gives some divide by zero errors, this is the laziest fix possible
    cc = corrcoef(target(:), output(:));
    if(numel(cc)>1) % Octave and MATLAB do corrcoef slightly differently, so this is to make things consistent
        cc = cc(2,1); 
    end
end