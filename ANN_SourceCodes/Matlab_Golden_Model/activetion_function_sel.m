function [ out] = activetion_function_sel( in, mode)
%ACTIVETION function
    switch mode
       case 'sigmond'       % sigmond
          out = (1./(1+exp(-(in))));
       case 'INT sigmond'       % ROUND TO INT sigmond 
          out = (1./(1+exp(-round(in))));
       case 'ReLU'       % ReLU
          out =  max(zeros(size(in)), in);%relu(in);
       case 'Leaky ReLU'       % Leaky ReLU
          out =  max(0.01*in, in);%relu(in);
       case 'SoftMax'       % SoftMax
          out =  (  exp(in)/sum(exp(in)) );
       case 'SoftMax INT'       % SoftMax
          out =  (  exp(round(in))/sum(exp(round(in))) );
       otherwise
          out = zeros(size(in))  
    end
end

