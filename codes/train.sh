echo "\n\n------- PreTraining Layer 1 -------\n\n"
th da.lua -nHidden 2500 -loader data_pretrain.t7 -save mae1 -optimization rmsprop
echo "\n\n------- Encoding Data -------------\n\n"
th encode.lua -loader preProcessed.t7 -model mae1/model.net -save encoded1.t7
echo "\n\n------- PreTraining Layer 2 -------\n\n"
th da.lua -nHidden 1000 -loader encoded1.t7 -save mae2 -optimization rmsprop
echo "\n\n------- Encoding Data -------------\n\n"
th encode.lua -loader encoded1.t7 -model mae2/model.net -save encoded2.t7 
echo "\n\n------- PreTraining Layer 3 -------\n\n"
th da.lua -nHidden 500 -loader encoded2.t7 -save mae3 -optimization rmsprop
echo "\n\n------- Encoding Data -------------\n\n"
th encode.lua -loader encoded2.t7 -model mae3/model.net -save encoded1.t7
