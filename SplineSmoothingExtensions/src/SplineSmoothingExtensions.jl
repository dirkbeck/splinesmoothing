module SplineSmoothingExtensions

using Pkg, Revise, SmoothingSplines, Gadfly, RDatasets
Pkg.activate(".")
using SmoothingSplinesExtensions

function get_loocv(X,Y,lambda)
    # gets leave-one-out cross-validation for a smoothingspline fit on given data
    err = 0.0;
    for i = 1:length(X);
      err += (Y[i] - predict(fit(SmoothingSpline, deleteat!(copy(X), i), deleteat!(copy(Y), i), lambda), X[i]))^2;
    end
    err = err/length(X);
end

function get_optimal_lambda(X,Y,lambda_precision,max_iter)
    # finds lambda such that loocv has a local minimum
    lambda = 0.0;
    cv_previous_lambda = Inf;
    cv_current_lambda = get_loocv(X,Y,lambda);
    while cv_current_lambda < cv_previous_lambda && lambda < max_iter*lambda_precision
        lambda += lambda_precision;
        cv_previous_lambda = cv_current_lambda;
        cv_current_lambda = get_loocv(X,Y,lambda);
    end
    return lambda - lambda_precision
end

function plot_lambda_vs_cv(X,Y,lambda_precision,iter)
    # plots lambda vs cv over the range [0 iter]
    lambdas = 0.0:lambda_precision:(iter-1)*lambda_precision;
    cvs = zeros(iter);
    for i=1:iter
        cvs[i] = get_loocv(X,Y,lambdas[i]);
    end
    plot(layer(x=lambdas, y=cvs, Geom.line),
        layer(x=[lambdas[findfirst(isequal(minimum(cvs)), cvs)]], y=[minimum(cvs)], Geom.point),
        Guide.xlabel("λ"),
        Guide.ylabel("LOOCV"),
        Guide.Title("Optimal λ"))
end

function get_error_bars(X,Y,lambda,confidence_interval)
    # returns upper and lower confidence interval using a leave-one-out bootstrap method.
    # 95% confidence_interval would return the 2.5th and 97.5th percentile of predictions at each X value.
    m = length(X);
    loo_estimates = zeros(m,m);
    lower_confidence_interval = zeros(m);
    upper_confidence_interval = zeros(m);
    for i=1:m
        X_loo_i_fit = fit(SmoothingSpline, deleteat!(copy(X), i), deleteat!(copy(Y), i), lambda);
        loo_estimates[i,:] = predict(X_loo_i_fit, X);
    end
    for i=1:m
        lower_confidence_interval[i] = sort!(copy(loo_estimates[:,i]))[Int(floor((m-1)*(1-confidence_interval)/2+1))];
        upper_confidence_interval[i] = sort!(copy(loo_estimates[:,i]))[Int(ceil((m-1)*(1 - (1 - confidence_interval)/2)+1))];
    end
    return lower_confidence_interval, upper_confidence_interval
end

function get_smoother_matrix(X, lambda)
    # gets the matrix that smoothens X in smoothing splines
    n = length(X);
    w = zeros(n,n);
    for i=1:n
        y = vec(zeros(n, 1));  # Equivalent to rep(0, length.out=n) but faster
        y[i] = 1;
        spl = fit(SmoothingSpline, X, y, lambda);
        w[:,i] = predict(spl);
    end
    return(w)
end

function get_boosting_smoothing_spline(X,Y,lambda,iter,v,plotMSEs)
    # gets Y predictions by boosting smoothing splines given number of iterations (iter) with a penalty scalar (v)
    # if plotMSES is set to TRUE, a plot of decreasing MSEs is displayed
    Ypred = copy(Y);
    for i=0:iter
        Ypred = predict(fit(SmoothingSpline, X, Y + v * (Y - Ypred), lambda));
    end
    return Ypred
end

function plot_boosting_smoothing_spline_MSEs(X,Y,lambda,iter,v)
    # similar to get_boosting_smoothing_spline, but plots the MSE at each iteration
    Ypred = copy(Y);
    mses = zeros(iter+1)
    for i=0:iter
        Ypred = predict(fit(SmoothingSpline, X, Y + v * (Y - Ypred), lambda));
        mse = 0.0;
        for j=1:length(Y)
            mse += (Ypred[j] - Y[j])^2
        end
        mses[i+1] = mse/length(Y);
    end
    plot(layer(x=0:iter, y=mses, Geom.line),
    layer(x=[findfirst(isequal(minimum(mses)), mses)-1], y=[minimum(mses)], Geom.point),
    Guide.xlabel("boosting iteration"),
    Guide.ylabel("MSE"),
    Guide.Title("Optimal # of Boosting Iterations"))
end

end
