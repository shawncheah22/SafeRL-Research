            +------------------------+
            |        ENVIRONMENT     |
            +-----------+------------+
                        |
                     (state)
                        |
                        v
           +--------------------------+
           |   Local Q-Network (θ)    |  --> predicts Q(s,a)
           +-----------+--------------+
                       |
     Action selection (ε-greedy)
                       |
                       v
                Take action in env
                       |
                  Get (r, s')
                       |
                       v
          +---------------------------+
          |  Target Q-Network (θ⁻)    |  --> computes y_t = r + γ * max Q(s', a')
          +---------------------------+
