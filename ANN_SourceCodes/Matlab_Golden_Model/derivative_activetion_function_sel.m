function [ derivative_out] = derivative_activetion_function_sel( in,out, mode)
%ACTIVETION function
    switch mode
       case 'sigmond'       % sigmond'
          derivative_out = (out .* (1-out));
       case 'INT sigmond'       % ROUND TO INT sigmond' 
          derivative_out = (out .* (1-out));
       case 'ReLU'       % ReLU'
          derivative_out = sign(sign(sign(out)-0.5)+1);% this is the simplest 1 line i could think of to make f(x)>=0 =>f'(x)=0
       case 'Leaky ReLU'       % Leaky ReLU'
           derivative_out = zeros(size(out));
            for i = 1:length(out)
                if (out(i) > 0)
                    derivative_out(i) = 1;
                else
                    derivative_out(i) = 0.01;
                end
            end 
       case 'SoftMax'       % SoftMax'
          derivative_out = (out .* (1-out));
       case 'SoftMax INT'       % SoftMax'
          derivative_out = (out .* (1-out));
       otherwise
          derivative_out = zeros(size(in)) 
    end
end

