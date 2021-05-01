import React from 'react';
// import classes from './Answer.module.css';
import Spinner from '../UI/Spinner.components.jsx';


const Answer = (props) => {
    const AnswerOrSpinnerOrNull = (props) => {
        if (props.loading === "spinner") {
            return (<Spinner />);
        }
        if (props.loading === "answer") {
            return (
                <div>
                    <p>Answer: {props.answer}</p>
                    <button onClick={() => {navigator.clipboard.writeText(props.answer)}}>
                        Click here to copy
                    </button>
                </div>
            );
        }
        return null;
    }
    return (
            <>
            {AnswerOrSpinnerOrNull(props)}
            </>
    );
}

export default Answer;
