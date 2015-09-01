function return_value = sum_of_square_error(ylist, tlist, N, num_output)

error = 0.0;
for n = 1:N,
    for k = 1:num_output,
        error = error + 0.5*(ylist(k,n) - tlist(k,n))*(ylist(k,n) - tlist(k,n));
    end
end

return_value = error;